# layered continuity-PPM — the kernel `ocean_continuity` actually wraps

**`do concurrent` is within ~10% of hand-written CUDA C at every production
config.** Measured, V100, nvfortran 26.5-0, `cuda_sync=2` (CUDA pays the same
per-kernel host sync `do concurrent` does, so the ratio reflects codegen):

```
 config (nxp x nyp x nz)      cells      dc_ms    cuda_ms  dc/cuda
 10km    (108x137x30)        489060     0.2699     0.1868    1.445
 0.1deg  (473x297x30)       4354110     1.0647     0.9645    1.104
 0.25deg (240x560x30)       4177080     1.0565     0.9621    1.098
 3km     (359x458x30)       5080800     1.2692     1.1729    1.082
 0.05deg (945x594x30)      17118000     4.1857     4.0475    1.034
 --- sensitivity to nz, same footprint ---
 0.1deg  (473x297x1)         145137     0.1960     0.1218    1.609
 0.1deg  (473x297x75)      10885275     2.4801     2.3290    1.065
```

Bit-identical throughout (`max |diff| = 0.0`, not merely <1e-12).

## Why this benchmark exists

The barotropic benchmark next door (`../continuity_ppm_benchmark`) measured the
**wrong kernel** for the regime that matters. Production splits continuity two ways
(`<scratch>/<user>/tassie_runs/gabight_acc_120d.*.log`, 0.1 deg, 34560 steps):

| profiler region | % (EPBL on) | % (EPBL off, est.) | wraps |
|---|---|---|---|
| `ocean_vmix_compute` | **39.1%** | — (absent) | EPBL + kappa-shear |
| `ocean_continuity` | 11.1% | **~20%** | `continuity_compute_fluxes` — **THIS kernel** |
| `ocean_barotropic_solver` | 8.6% | ~16% | `barotropic_substep_nonlinear` — the *other* one |

With EPBL and kappa-shear off, vmix's 39% leaves the budget and continuity becomes
co-dominant with `ale_remap`. That is the regime this kernel is worth optimising for.

## The finding: the gap is a function of TOTAL CELLS, not dimensionality

The `nz=1` row is the control. At 145k cells it reads **1.609x** — matching the
*barotropic* kernel's 1.60x at 140k cells almost exactly, despite being a different
routine with 11 loops instead of 9. Same mechanism, and it is the one the barotropic
README documents: a fixed ~10 us of dead host time per loop, which dominates when
there is little work per launch and vanishes when there is a lot.

**The layers are what rescue this kernel.** 30 of them turn a 145k-cell problem into
a 4.35M-cell one, and the per-launch cost amortises away: 1.61x -> 1.10x. Push to
0.05 deg (17.1M cells) and it is 1.03x — indistinguishable from CUDA C.

Both compiler defects are absent, as in the barotropic kernel and for the same
reasons: arrays are derived-type *components* (`ms%h_layer`), not `grid%`-bounded
dummies (no bug 1 — all 11 loops report `auto-collapsed`, `collapse(3)` on the five
3-D loops and `collapse(2)` on the boundary loops); and the caller hoists every read
into a scalar before calling the PPM helpers (no bug 2).

## What this means for the CUDA question

At 4.35M cells `do concurrent` is already at **90% of hand-written CUDA**. Rewriting
this kernel in CUDA C would win ~10% of a region worth ~20% of runtime with EPBL off
= **~2% of total**, in exchange for a permanently hand-maintained C port of a PPM
scheme. The arithmetic does not support it.

"Continuity is the bottleneck" and "continuity has a big compiler-attributable gap"
are different claims. The first is true with EPBL off; the second is not — for *this*
kernel. The headroom that does exist is in the **barotropic** kernel (1.6x, but a
smaller slice) and it is launch overhead, recoverable by fusing loops rather than by
changing language. See `../continuity_ppm_benchmark/README.md`.

**Exception: the 10 km config (108x137x30, 489k cells) reads 1.445x.** Small domains
do not get the amortisation. If that config matters, it is the one to look at.

## Is the OpenACC runtime taxing the CUDA numbers? No — measured, it is a tie.

Every CUDA figure above was produced by launching the kernel *from Fortran*, with device
pointers obtained through `!$acc host_data use_device(...)` — a present-table lookup, per
array, per call, **fourteen arrays every rep**. If that cost anything, the CUDA column
would be understated and the `dc/cuda` ratios would be too *low*.

`layered_native.cu` settles it: `cudaMalloc` + `main()`, no Fortran, no OpenACC, no
nvfortran runtime. It calls the **same `continuity_layered_cuda_launch()` from the same
`layered_kernel.o`** that `layered_bench` links — the kernel is not duplicated, so the two
paths cannot drift apart.

```
  CUDA via Fortran driver (host_data)  : 0.9481 ms   <- the existing number
  CUDA native (cudaMalloc, no OpenACC) : 0.9475 ms
  ratio                                : 1.001 x     <- no measurable OpenACC tax
```
473x297x30, `cuda_sync=2`, mean/rep, best of ~10 paired runs in an idle window.
**PROVISIONAL — taken on a contended V100** (see caveat below). The two are
indistinguishable: 0.3% apart at the median, against a run-to-run spread larger than that.

**So the repo's CUDA numbers are already fair, and no conclusion shifts.** `host_data` is
not silently handicapping CUDA; the ~10% edge at 4.35M cells is real codegen, and
`ocean_continuity` stays "nothing to fix". This closes a loophole rather than opening one.

### How correctness was verified — bit-identity, and it is contention-proof

`./layered_bench <args> 1` dumps `fh_ref.bin`: its **inputs** (`h`, `u`, `v`) as well as
both outputs (`do concurrent`, and CUDA-via-`host_data`). `layered_native` then asks two
*different* questions, because conflating them convicts a clean port of a bug:

1. **Own init.** Does the native driver's own initialisation reproduce the Fortran's?
   To ~1 ulp, yes (`max rel` 2.2e-16 on `h`) — but *not* bit-exactly, and it cannot be:
   the init calls `sin`/`cos`/`exp`, nvfortran resolves those to **libpgmath** and nvcc's
   host pass to **glibc's libm**, and the two disagree in the last bit. Propagated through
   the kernel that gives `max|diff|` 1.8e-15 on a field of range ±1.2 (1.5e-15 of field
   scale). Note `flux_h` is a **divergence**, so cells where the fluxes cancel have
   `fh ≈ 0` and *pointwise relative* error there is meaningless — judged by that bar, a
   1-ulp input wobble reads as a 1.8e-8 "failure". It is scored against field magnitude.
2. **Fortran's init uploaded.** Same inputs, same kernel, only the *allocator* differs.
   This removes the libm variable and leaves exactly one, which is the one being tested:

```
    flux_h vs do concurrent   max|diff| 0.0000e+00   BIT-IDENTICAL
    flux_h vs CUDA/host_data  max|diff| 0.0000e+00   BIT-IDENTICAL
```

Test (2) is the load-bearing one and it passes at **every** config tried — `473x297x30`,
`473x297x1`, `108x137x30`, `240x560x30`, and a deliberately asymmetric `37x61x7` that
would expose any T/U/V stride swap or misplaced ghost wall. Exactly 0.0, not "small".
That also *confirms* the libm hypothesis in (1) rather than asserting it: removing the
input difference removes the output difference entirely.

**Correctness results are unaffected by GPU contention** — these can be relied on.

### ⚠ The timings above are contended and need a serial re-run

Up to 9 sibling agents shared this one V100 during measurement. Observed swings were
brutal — the same binary read 0.95 ms and 38 ms minutes apart, and a naive
"min over trials" during the worst window produced a **spurious 7.1x** simply because the
Fortran binary never got a clean slot while the native one did. Min-of-trials is *not*
robust when contention is sustained. The 1.001x above comes from a window where
`nvidia-smi --query-compute-apps` showed the GPU genuinely idle and both binaries
clustered tightly (Fortran 0.948–0.979, native 0.948–0.958 across ~10 runs each). It
should be re-taken serially before being quoted.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run                          # 473 297 30 (the 0.1 deg config)
./layered_bench 945 594 30                # [nx_phys] [ny_phys] [nz]
./layered_bench 473 297 30 500 20         # [nreps] [nwarm]
./layered_bench 473 297 30 200 10 0       # [cuda_sync] 0=async 1=per-rep 2=per-kernel
make collapse                             # per-loop -Minfo
make regs                                 # LDG + spills per kernel

make native && ./layered_native           # no-Fortran, no-OpenACC driver
make run-native                           # dump fh_ref.bin, then verify against it
./layered_bench 473 297 30 200 10 2 1     # [dump] 1 -> write fh_ref.bin
```

`layered_native` takes the **same six arguments** as `layered_bench`, with the same
defaults (`473 297 30 200 10 2`). It verifies automatically when an `fh_ref.bin` for the
matching domain is present, and skips (loudly, with the command to produce one) otherwise.

`nx_phys`/`ny_phys` are **interior** cells; `nx_total = nx_phys + 2*nghost` (nghost=3).
Unlike the barotropic kernel, this one is **ghost-aware** — it zeroes mass flux at the
*physical* edges (`nghost+1`, `nghost+nx_phys+1`), so that relationship must hold or
the walls land in the wrong place.

## Files

```
continuity_layered.F90  production extract, VERBATIM (continuity.F90:612-775
                            + the 4 PPM helpers) + MRE stubs for metrics / multilayer
                            state / scratch / continuity_t. Only scaffolding change:
                            the helpers are public so controls can reuse them.
layered_kernel.cu           faithful CUDA C port. 11 kernels, one per Fortran loop,
                            same order. Three stride classes (T/U/V) — mixing them is
                            silent corruption, not a compile error.
layered_bench.F90           driver. mem:separate + manual deep copy; both toolchains
                            read the SAME device allocation via host_data use_device.
                            Optional 7th arg dumps fh_ref.bin for layered_native.
layered_native.cu           NATIVE driver: cudaMalloc + main(), no Fortran, no OpenACC.
                            Links the SAME layered_kernel.o as layered_bench -- the
                            kernel is NOT duplicated, so the two CUDA paths cannot
                            drift. Isolates the cost of the host_data launch path.
```

## Caveats

- **All-wet** (`wet_T = 1`), so `ppm_mirror_h` and the slope-flatten multiply by 1 and
  the land branches never fire. A coast would exercise divergence. Untested.
- **`use_ppm_limit_pos = .false.`** (production default) — `ppm_limit_pos` never runs.
- The kernel measures **1.07 ms/rep** at the 0.1 deg config against production's
  2.16 ms/call for `ocean_continuity`. Not a discrepancy: that timer region also
  covers the Fox-Kemper fold and the tracer-advect cadence, not just this routine.
- The CUDA port is **faithful, not tuned**: no tiling, no `__ldg`, no `launch_bounds`.
  A tuned port would likely widen the gap — a different question from "does
  nvfortran's codegen keep up?".
- No FLAT or ACC control here (unlike the barotropic bench). The barotropic result
  said flattening loses and the OpenACC wrapper spills; nothing suggests this kernel
  would differ, but that is an assumption, not a measurement.
