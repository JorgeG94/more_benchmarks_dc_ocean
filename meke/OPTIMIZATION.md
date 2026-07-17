# Optimizing the MEKE-step CUDA kernel

A best-CUDA-practices pass over `meke_kernel.cu` (the faithful 16-kernel
transliteration of the `do concurrent` MEKE step), freed from the faithfulness
rule but held **bit-identical** to the faithful launcher. `ab_main.cu` checks
`max|diff| == 0` on the primary output field `meke`. V100, gabight config
(473×297×30, degenerate nml as in `meke_bench.F90`).

## Result

**0.239 → 0.188 ms/step, 1.28× (bit-identical).** 1.21–1.90× across sizes — much
more on smaller grids, where cutting 16 launches to 3 dominates.

| size | faithful ms | opt ms | speedup | max\|diff\| |
|---|---|---|---|---|
| 108×137×30 | 0.0818 | 0.0430 | **1.90×** | 0 |
| 473×297×30 | 0.2393 | 0.1876 | **1.28×** | 0 |
| 473×297×50 | 0.2950 | 0.2427 | **1.22×** | 0 |
| 945×594×30 | 0.8125 | 0.6720 | **1.21×** | 0 |

## What worked (the default build, `OPTVER=1`)

Three changes, all *removing genuine waste* rather than being clever:

1. **Launch fusion, 16 kernels → 3.** The step has exactly three barriers:
   a per-cell pass A (`mass / length / source / drag`), a per-cell pass B
   (`flux-divergence / drag / kh / ku`), and the `sn_u`/`sn_v` zeroing that
   pass A reads. `meke_kernel.cu`'s own `fused` path already collapses to 6;
   this goes to **3** by merging the two sn-zero launches into one strided
   kernel and folding the two flux launches + the divergence launch into
   pass B (see #2). Fewer launches is the whole game on small grids (1.90×).
2. **Eliminate the `uflux`/`vflux` global arrays.** Pass B recomputes the 4
   surrounding face fluxes **in registers** from the post-drag energy
   (`meke_A`) + `mass_ws` stencils, instead of a producer kernel writing
   `nu+nv` doubles that the divergence kernel reads back. The MEKE flux is
   cheap (a harmonic mean + a few mults), so the 2× redundant flux math costs
   far less than the round-trip it removes — the opposite of the continuity
   kernel, where recompute of the expensive PPM *lost*.
3. **32-bit indexing.** `nx*ny*nz < 2³¹` for every realistic MEKE grid
   (479×303×30 ≈ 4.35e6 ≪ 2.1e9), so `meke_args.h`'s `size_t` address math is
   pure overhead. Overridden locally in `opt_kernel.cu`.

**Double-buffer.** Pass A reads `meke`(old) and writes a scratch `meke_A`
(post-drag); pass B reads the `meke_A` 5-point stencil and writes the final
`meke`. One extra `nt`-sized scratch, but it removes the `nu+nv` flux arrays
(net device memory *down*) and gives the flux a race-free neighbourhood
(`meke_A` is finished at the kernel boundary).

## What was tried and LOST / not taken

| idea | where | result | why |
|---|---|---|---|
| staged flux (keep `uflux`/`vflux`, 4 kernels) | `OPTVER=2` | 1.23× (vs 1.28×) | the flux round-trip it keeps costs more than the recompute it avoids; kept as the general fallback (below) |
| fewer than 3 kernels | — | impossible | pass A → flux → pass B are three genuine data-dependency barriers; nothing collapses them without a device-wide sync |

## Caveats

- **Config assumption in the recompute path.** Pass B's flux recompute reads
  `kh_diff` while pass B also *writes* `kh_diff` — a read/write hazard in
  general. It is benign here because the nml sets `khmeke_fac = 0`, so the
  `kh_diff` term is `0.0 * (finite) == 0.0` regardless of which copy is read,
  and `kh_u` collapses to `fmax(0, kh_bg)` exactly. **With `khmeke_fac ≠ 0`,
  use `OPTVER=2`** (staged flux in its own kernel), which is general and only
  ~4% slower. Every other input pass B reads (`meke_A`, `mass_ws`, the per-cell
  factors) is produced by pass A, so it is hazard-free by construction.
- **Conservative timing.** Both launchers `cudaDeviceSynchronize()` once per
  step (matching `meke_cuda_launch`), so the speedup isolates launch-count /
  memory-traffic, *not* a synchronization difference. Removing the per-step
  sync (letting the optimized launcher pipeline across steps) is an additional
  win not counted here.
- Degenerate gabight config (`a_* = 0`, `cd_scale = 0`) exercises MEKE's
  *shape*, not every arithmetic branch — same limitation as `meke_bench.F90`.
  The fusion is structural, so it holds regardless, but the exact-zero
  agreement should not be read as general nvcc FMA-grouping proof.

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu meke_kernel.cu -O3 -arch=sm_70 -I../common -I. -o ab
# through the shared-GPU lock only:
bash ../tmp_local_artifacts/gpu_run.sh meke ./ab 473 297 30 300   # max|diff| + timing
nvcc -DOPTVER=2 ... -o ab            # the general staged-flux variant
```

`opt_kernel.cu` is a standalone best-CUDA artifact; the faithful
`meke_kernel.cu` stays untouched as the codegen-comparison baseline. Wiring
`meke_opt_launch(const MekeArgs*, double *meke_scratch)` into a driver is a
small follow-up (it needs one extra `nt`-sized scratch and drops the
`uflux`/`vflux` allocations).

## DC (do concurrent) optimization

The portable analogue of the CUDA win: the *fusion* carries over to `do
concurrent`, the *async* does not. `meke_step_ext_fused` (in `meke.F90`, added
alongside the untouched faithful `meke_step_ext`) collapses the **16 same-bounds
`do concurrent` loops to 6** by fusing loops that share iteration bounds and have
no cross-loop halo (neighbour) dependency. Driver `drivers/dc_ab.F90`, target
`make dcab` (binary `dc_ab_$(DATA)`), mirrors continuity's A/B harness: seed →
faithful warm+timed → snapshot `meke` → fused warm+timed → `max rel < 1e-12`
compare + ms/rep + speedup.

**What fused (which faithful loops → one DC loop):**

| fused loop | faithful loops | bounds | why safe |
|---|---|---|---|
| pass A | 1 rd-zero, 4 mass, 5 length-scales, 6 ke_diss stage, 7 source, 8 drag¹ | `nx×ny` | every member reads/writes only cell `(i,j)` of what an earlier member wrote |
| u-flux | 9 zero + 10 interior | `(nx+1)×ny` | interior fill guarded (`i∈[2,nx]`), else 0 — bit-identical to zero-then-fill |
| v-flux | 11 zero + 12 interior | `nx×(ny+1)` | same |
| pass B | 13 divergence, 14 drag², 15 kh, 16 ku | `nx×ny` | flux stencil read from the prior loops (across the DC barrier); only cell `(i,j)` touched within |

The two `sn_u`/`sn_v` face zeroings (loops 2,3) stay separate — different bounds.
The flux loops are deliberately **not** folded into pass B: that would need `meke`
double-buffering plus a `kh_diff` read-after-write hazard (the same over-fusion
the CUDA path absorbed with a scratch buffer + a `khmeke_fac=0` caveat). Kept
portable and unconditionally exact here. Any config outside the gabight MEKE-on
path (`k4≥0`, `kh<0`, or `khmeke_fac≠0`) falls back to the faithful step, so the
fused routine is bit-identical to `meke_step_ext` for **all** inputs.

**Result (V100, `-stdpar=gpu -acc=gpu`, DC A/B, `max rel diff = 0`):**

| grid | faithful ms/rep | fused ms/rep | speedup |
|---|---|---|---|
| 108×137×30 | 0.1916 | 0.1283 | **1.49×** |
| 473×297×30 | 0.3624 | 0.3060 | **1.18×** |

Bit-identical (`max|diff| = 0`, `max rel = 0`), non-zero output both sides. As on
CUDA, the win is launch-count-bound so it grows on smaller grids. **CPU-portable:**
`make dcab DATA=none && ./dc_ab_none 108 137 30 20 5` builds with plain
`-stdpar=multicore` (no CUDA/GPU, no lock) and is bit-identical (`max rel 0`);
the fused form is slower on the CPU (~0.64×) because multicore has no
per-loop launch overhead to amortise — the fusion buys nothing there, but costs
nothing in correctness.

Reproduce:

```bash
make dcab DATA=acc
bash ../tmp_local_artifacts/gpu_run.sh meke-dcab ./dc_ab_acc 473 297 30 200 20
make dcab DATA=none && ./dc_ab_none 108 137 30 20 5   # CPU bit-identity, no lock
```

## Head-to-head: opt-CUDA vs opt-DC (shared host_data driver)

The two results above come from **two separate harnesses** — `dc_ab` (optimized
vs faithful *do concurrent*) and the `ab` CUDA A/B — each timing its own memory
image. That leaves a caveat: the opt-CUDA `0.188 ms` and the opt-DC fused
`0.306 ms` were never measured side-by-side on the same arrays in the same
process, so a systematic per-harness difference (allocation provenance, warm
state, clock) could bias the cross-toolchain ratio.

`cmp_acc` (`make cmp`, binary `cmp_acc`) removes it. **One binary, one device
image.** It sets up the state exactly as `dc_ab`, `enter_data`s it once, then:

1. reseeds `meke` to the seed, times `meke_step_ext_fused` (opt-DC), snapshots `meke`;
2. reseeds `meke` to the **identical** bits, times `meke_opt_launch` (opt-CUDA)
   **on the same device arrays**, handed over inside a single `!$acc host_data
   use_device(...)` through a flat `bind(C)` bridge (`meke_opt_launch_flat`);
3. compares the two `meke` fields (`< 1e-12` rel bar) and prints both ms/rep + the ratio.

The bridge (`drivers/meke_bridge.F90`) is the redi pattern: a flat wrapper
(`meke_opt_launch_flat` in `opt_kernel.cu`) explodes the `MekeArgs` struct into
one plain argument per device pointer + scalar, so the Fortran side hands each
array over uniformly. **The bridge is compiled WITHOUT `-cuda`** (it declares
plain host arrays; `-cuda` would type-check them as device → `NVFORTRAN-S-0528`);
`-cuda` is on the **link line only**. `meke_opt_launch`'s pass-A/pass-B scratch
is a driver-owned `nx×ny` Fortran array, `enter_data`'d and passed via `host_data`
like the rest.

**Result (V100, 473×297×30, one shared image, DATA=acc):**

| | ms/rep | agreement |
|---|---|---|
| opt-DC (`meke_step_ext_fused`) | ~0.302 | — |
| opt-CUDA (`meke_opt_launch`)   | ~0.192 | — |
| **ratio** | **opt-CUDA 1.57× faster** | `max rel diff = 0.0` (bit-identical) |

Both legs on one memory image, `meke` reseeded to identical bits between them,
output non-zero (`[1.0e-3, 1.1e-2]`). The exact-zero agreement is the same
degenerate-config caveat as everywhere else (`a_* = 0`, `khmeke_fac = 0`), so
read it as *shape* agreement, not general nvcc-vs-nvfortran FMA proof — but the
**1.57× on one truth confirms the two-harness ratio was not an artefact**:
opt-CUDA's async/launch-count edge over portable `do concurrent` is real and
survives being put on a single memory image.

Reproduce:

```bash
make cmp
bash ../tmp_local_artifacts/gpu_run.sh meke-cmp ./cmp_acc 473 297 30 200 20
```
