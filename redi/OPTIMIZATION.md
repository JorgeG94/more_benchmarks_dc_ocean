# Optimizing the Redi apply-flux CUDA kernel

A best-CUDA-practices pass over the faithful port (`redi_kernel.cu`), freed from
the "faithful transliteration" rule but held **bit-identical** to it
(`ab_main.cu` checks `max|diff| == 0` on both tracers, not just <1e-12). Redi is
the biggest kernel in the model (~40%), so this is the highest-value target.
V100, 473×297×30, `NZ_STACK_MAX=128` (production).

## The optimization — precompute hoist in `redi_apply_flux`

The faithful `kRediApplyFlux` visits every T-cell `(i,j)` and evaluates **four**
face fluxes (W/E/S/N). Each face flux reconstructs its **left and right** tracer
column from scratch (`redi_tracer_column`: PLM slopes + PPM edges + per-layer
limiter over the whole column). That is **8 column rebuilds per cell** — but
only **5 distinct columns** are involved:

```
        (i,j)   <- W-right, E-left, S-right, N-left   (used by ALL FOUR faces)
        (i-1,j) <- W-left        (i+1,j) <- E-right
        (i,j-1) <- S-left        (i,j+1) <- N-right
```

The **centre column `(i,j)` is rebuilt 4×**. `kOptApplyFlux` reconstructs it
**once** into registers/local, reuses it in all four faces, and reconstructs
only the distinct neighbour per face → **8 rebuilds → 5**. This is the exact
analogue of the continuity fusion win and the lever the README calls out.

`redi_tracer_column` is a deterministic pure function of the column data, so
computing it once vs four times yields **the same bits**; the W,E,S,N order and
the `ks=1..ns-1` accumulation order into `dTr` are preserved exactly. Hence the
result is **bit-exact**, not merely FMA-level.

Secondary: 32-bit indexing (`IDX32`, default on) — the domain fits int32
((nx+1)·ny·nsurf ≈ 9.0e6 ≪ 2³¹). FP-neutral, so bit-identicality holds.

## Result — bit-identical, across sizes

| grid | faithful ms/call | optimized ms/call | speedup | max\|diff\| |
|---|---|---|---|---|
| 236×149×30 | 10.87 | 9.36 | **1.16×** | 0 |
| 473×297×30 | 44.98 | 42.11 | **1.07×** | 0 |
| 473×297×50 | 75.44 | 70.53 | **1.07×** | 0 |

Reproduce:
```bash
nvcc ab_main.cu opt_kernel.cu redi_kernel.cu -O3 -arch=sm_70 -I../common \
     -DMODEL_NZ_STACK_MAX=128 -o ab
# GPU runs go THROUGH THE LOCK on the shared V100:
bash ../tmp_local_artifacts/gpu_run.sh redi ./ab 473 297 30 50 10
```

## Why the whole-launch number is "only" ~1.07×, not the ~1.35× the lever quotes

Per-kernel profile (nsys, one launch = CalcCoeffsX + CalcCoeffsY + 2×(Snapshot +
ApplyFlux)), 473×297×30:

| kernel | faithful ms | optimized ms | share of launch |
|---|---|---|---|
| ApplyFlux (×2) | 13.78 each | **12.17 each** | 61% |
| CalcCoeffsX | 8.70 | 8.80 | 19% |
| CalcCoeffsY | 8.68 | 8.75 | 19% |
| Snapshot/FaceCopy | ~0.09 | ~0.09 | <1% |

The hoist targets ApplyFlux and speeds **that kernel 1.13×** (13.78→12.17 ms).
Two facts dilute it to 1.07× at the launch level:

1. **ApplyFlux is 61% of the launch, not 100%.** CalcCoeffs (the neutral-position
   search) is the other 38% and the hoist does not touch it.
2. **Recon is only ~31% of ApplyFlux; the per-face sublayer sweep is the rest.**
   Eliminating 3 of 8 rebuilds removes ~1.6 ms of ~4.3 ms of recon. The
   `ks`-loop over `redi_sublayer_dT` / `redi_ppm_ave` (4 faces × ns−1 surfaces)
   is **genuine, non-redundant** work — nothing to hoist there bit-identically.

The ~1.35× the README quotes is the kernel/algorithm-level figure (and the
Fortran `do concurrent`, where apply-flux dominates more). On the GPU the flux
sweep, not the recon, is the apply-flux hot path, so the same lever buys less.

## Bonus the hoist bought for free (ptxas `-v`, sm_70)

| | registers | stack frame | spills | occupancy (reg-limited) |
|---|---|---|---|---|
| faithful ApplyFlux | 93 | 16432 B | 0 | ~34% (704 thr/SM) |
| **opt ApplyFlux** | **64** | **14384 B** | 0 | **~50% (1024 thr/SM)** |

Reconstructing the centre once collapses the redundant recon live-ranges: 29
fewer registers and 2 KB less local memory, lifting reg-limited occupancy from
~34% to ~50% with no new spills. Part of the 1.13× is this occupancy gain, not
just the removed FLOPs.

## Tried and LOST / rejected (kept off the shared GPU after analysis)

| idea | verdict | why |
|---|---|---|
| 32-bit indexing (`IDX32=0` vs `1`) | **neutral** (within run noise, 41–42 ms) | ApplyFlux is local-memory/compute-bound, not address-bound; the register drop above came from the recon hoist, not int32. Kept on (free, harmless). |
| Global precompute-once of tracer columns to scratch (1 recon/cell in a separate kernel, ApplyFlux just reads) | rejected by cost model | Removes the last 4 redundant recons (5→1) but adds ~140 MB write + ~5× read of 4 per-cell recon arrays. Est. net ≈ break-even (recon is only ~22% of the hoisted ApplyFlux); the added global traffic eats the saved FLOPs. Not worth the complexity/risk on a shared V100. |
| Carry `tlb→tlt`, `trb→trt` between consecutive `ks` in the sublayer loop | rejected | The bottom interface of surface `ks` equals the top of `ks+1`, so 2 of 4 interface interpolations repeat — but they must then be computed even for `hEff==0` surfaces the loop currently skips, and they are a small part of `redi_sublayer_dT` (the two `redi_ppm_ave` calls dominate). Marginal, and fragile for bit-identicality. |
| Fuse CalcCoeffsX+Y, or Snapshot into ApplyFlux | rejected | Only saves launch overhead (µs) against 8–14 ms kernels; no work removed. |
| Hoist the redundant `build_column` in CalcCoeffs (each cell's column is built 4× across adjacent u/v-faces) | not attempted | The redundancy is **cross-thread** (adjacent faces in different threads), unreachable in the one-thread-per-face layout without a global precompute pass — a much larger rewrite, and the neutral-position search (the actual CalcCoeffs hot path) is per-face and non-shareable. |

## Files

- `opt_kernel.cu` — optimized launcher `redi_opt_launch` + `kOptApplyFlux` (the
  hoist). All device helpers and Phase-A kernels are transcribed verbatim from
  `redi_kernel.cu` (renamed / anonymous-namespace so both objects link);
  **only apply-flux carries the optimization**.
- `ab_main.cu` — A/B harness (init mirrors `drivers/cpp_main.cu`): faithful vs
  optimized, diffs both tracers (bit-exact check), times both with CUDA events.
- `redi_kernel.cu` — untouched faithful baseline.

## DC (do concurrent) optimization

The **same precompute hoist**, applied to the portable-Fortran `do concurrent`
side (`ocean_redi.F90`) — this is the lever `NOTES_ON_PERF.md` quotes at ~1.35×,
and the reason it says `do concurrent` already wins on this kernel.

**What was hoisted.** The faithful `redi_apply_flux_impl` loop visits every
T-cell `(i,j)` and calls `redi_face_flux` on each of its four bounding faces
(W/E/S/N). `redi_face_flux` rebuilds **both** its left and right tracer column
from scratch via `redi_tracer_column` (PLM slopes + PPM edges + per-layer
limiter + the `interface_scalar` solve over the whole column) — **8 column
rebuilds per cell**, but only **5 distinct columns** exist: the centre `(i,j)`
is a column of *all four* faces and is rebuilt **4×**; each of the 4 neighbours
once.

The new **`redi_apply_flux_hoist`** (added ALONGSIDE the byte-identical faithful
path; the faithful routine is untouched) reconstructs the centre column **once**
per cell into a frame-local (`redi_cell_flux`) and reuses it across all four
faces via `redi_face_flux_hoist`, which rebuilds only the distinct neighbour
column → **8 rebuilds → 5**. The heavy column arrays live in `redi_cell_flux`'s
frame exactly as they did in `redi_face_flux`, so the `do concurrent` loop body's
footprint stays `local(k, dTr, iaij)` (no new DC-locals — avoids the gfortran
dc-local corruption the faithful path already sidesteps).

**Bit-exact, not just FMA-level.** `redi_tracer_column` is a deterministic pure
function of the read-only `(h, hTr_in)` column, so computing the centre once vs
four times yields **the same bits**; the W,E,S,N face order, the L/R argument
order into `redi_sublayer_dT`, and the `ks=1..ns-1` accumulation order into `dTr`
(with the `+KoL`/`−KoR` sign rule) are preserved verbatim. Nothing is
reassociated.

**Result** (`drivers/dc_ab.F90`, `make dcab`; apply-flux only, coefficients held
fixed — the DC analogue of the CUDA *ApplyFlux* row, since only apply-flux
changed):

| grid | DATA | faithful ms/rep | hoisted ms/rep | speedup | max rel diff |
|---|---|---|---|---|---|
| 473×297×30 | acc (V100) | 28.32 | 22.80 | **1.24×** | 0 |
| 80×60×10 | none (CPU) | 2.47 | 1.91 | **1.29×** | 0 |

The DC number (~1.24–1.29×) is far above the CUDA whole-launch 1.07× and near the
CUDA kernel-level 1.13× / the README's ~1.35× because on the `do concurrent` side
apply-flux **is** the kernel here (coeffs excluded) and the redundant recon is a
larger share of it — the exact effect `NOTES_ON_PERF.md` predicts.

**Portability confirmed.** `make dcab DATA=none && ./dc_ab_none 80 60 10 5 2`
builds and runs on the CPU with **no CUDA**, reports `max|diff| = 0` (still
bit-identical), and non-zero output — valid F2018 `do concurrent … local(…)`.

Reproduce:
```bash
make dcab                    # DATA=acc (GPU) by default
bash ../tmp_local_artifacts/gpu_run.sh redi-dcab ./dc_ab_acc 473 297 30 200 10
make dcab DATA=none && ./dc_ab_none 80 60 10 5 2     # CPU, no lock needed
```

### DC files

- `ocean_redi.F90` — adds `redi_apply_flux_hoist` (+ `redi_apply_flux_hoist_impl`,
  `redi_cell_flux`, `redi_face_flux_hoist`) after the verbatim faithful path;
  faithful `redi_apply_flux`/`_impl`/`redi_face_flux` unmodified.
- `drivers/dc_ab.F90` — A/B driver (models `continuity_layered/drivers/dc_ab.F90`):
  Phase A once, then faithful vs hoisted apply-flux — bit-identity check on hTr
  plus ms/rep and speedup.

## Head-to-head: opt-CUDA vs opt-DC (shared host_data driver)

The `dc_ab` (opt-DC) and the CUDA `ab` (opt-CUDA) numbers above were measured in
**separate binaries** on **separate device allocations** — opt-DC on OpenACC
`enter data`, opt-CUDA on the native-driver's `cudaMalloc`. That leaves a
two-harness caveat: are we comparing the two kernels, or two allocators/harnesses?

`drivers/cmp_main.F90` (`make cmp` → `cmp_acc`) closes it. **One binary** times
the **optimized do-concurrent** path and the **optimized CUDA** launcher on the
**SAME OpenACC device arrays**: the bridge `drivers/redi_bridge.F90` wraps
`redi_opt_launch` (opt_kernel.cu) in `!$acc host_data use_device(...)`, handing
the CUDA kernels the exact allocations the `do concurrent` variant reads. No
copies, no second allocator — both toolchains read one truth, agreement is
checked in-process.

**Region timed (identical on both sides — Phase A + Phase B):**

| side | what runs per rep |
|---|---|
| opt-DC | `redi_calc_coeffs` (faithful Phase A) + `redi_apply_flux_hoist` (hoisted Phase B: face copy + per tracer snapshot + hoisted apply-flux) |
| opt-CUDA | `redi_opt_launch`: `kOptCalcCoeffsX/Y` + `kOptFaceCopy` ×2 + per tracer (`kOptSnapshot` + `kOptApplyFlux`) |

Each side owns distinct output state (`ms`/`redi` vs `ms_cu`/`redi_cu`) built from
the identical initial condition and run for the identical rep count, so both
accumulate the same non-idempotent apply-flux steps and their final states must
agree bit-for-bit.

**Result** (V100, 473×297×30, `NZ_STACK_MAX=128`, 200 reps / 10 warm, two runs):

| side | ms/rep | ratio | agreement (max rel) |
|---|---|---|---|
| **opt-DC** | **39.96 – 40.18** | — | — |
| opt-CUDA | 42.73 – 43.01 | opt-CUDA/opt-DC = **1.07× (opt-DC faster)** | **3.13e-15** (< 1e-12) |

Output is non-zero (hTr range 282 … 4687), so the agreement check is non-trivial;
`max rel = 3.1e-15` is FMA-contraction level, confirming opt-DC and opt-CUDA are
bit-identical on **shared** device memory. This reproduces the separate-harness
verdict — opt-DC ≈ opt-CUDA, DC marginally ahead — with the allocator/harness
variable eliminated, and matches the whole-launch CUDA figure (~42 ms) from the
`ab` table above. `do concurrent` already wins on redi; the shared driver proves
it is not a harness artifact.

Reproduce (GPU run goes THROUGH THE LOCK on the shared V100):
```bash
make cmp
../tmp_local_artifacts/gpu_run.sh redi-cmp ./cmp_acc 473 297 30 200 10
# or: make run-cmp ARGS="473 297 30 200 10"
```

### cmp files

- `drivers/redi_bridge.F90` — `!$acc host_data use_device` bridge to
  `redi_opt_launch`. **Must NOT be compiled with `-cuda`** (LINK-ONLY: `-cuda`
  flips nvfortran to CUDA-Fortran mode → NVFORTRAN-S-0528 device-attribute
  mismatch on the host_data arrays). Mirrors legacy `redi_cuda.F90`, bind(C)
  name changed to the bare `redi_opt_launch`.
- `drivers/cmp_main.F90` — the shared driver: init, `enter data`, times opt-DC
  then opt-CUDA over identical reps, copies both back, checks agreement, prints
  both ms/rep + ratio + max rel diff.
- Makefile targets `cmp` / `run-cmp`; mixed nvfortran(no `-cuda`)+nvcc compile,
  `-cuda` at link only.
