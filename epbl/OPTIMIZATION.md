# Optimizing the EPBL column kernel

An optimized CUDA pass over `epbl_kernel.cu`'s faithful port (`epbl_faithful`,
variant 0). EPBL is a **single monolithic iterative column solver** — one thread
per column, a chaotic MLD root-find inside. There is nothing to fuse; the only
lever is the launch configuration, which is exactly the knob `do concurrent`
cannot reach. V100, 473×297×30 (real gabight bathymetry, 89.3% wet).

## Result

**6.77 → 4.92 ms/rep, 1.375× — kd_int bit-identical, MLD bit-identical.**

| metric | faithful | optimized |
|---|---|---|
| kernel | `epbl_faithful` | `epbl_opt` |
| registers/thread | 184 | 128 (`__launch_bounds__` cap) |
| occupancy limit (blocks/SM) | 2 | 4 |
| achieved occupancy (ncu) | 8.36 % | 17.28 % |
| ms/rep | 6.77 | 4.92 → **1.375×** |

The occupancy roughly **doubles** (2 blocks → 4 blocks/SM; 8.36 % → 17.28 %
achieved), and the runtime falls by very nearly the same factor — this kernel is
occupancy/latency-limited, so more resident warps directly hides the long
dependent-chain latency of the column sweep.

## What worked — two changes, no arithmetic touched

`opt_kernel.cu` copies the `epbl_column_body` float arithmetic **verbatim**
(same statements, order, branches, FMA-contractible expressions) and changes
only:

1. **`__launch_bounds__(128, 4)` — THE win.** The faithful port needs 184
   registers, so ptxas fits only 2 blocks (256 threads = 8 warps) per SM →
   ~12.5 % theoretical / 8.36 % achieved occupancy. Capping the budget to 128
   registers lets 4 blocks (512 threads = 16 warps) reside → 25 % / 17.28 %.
   `do concurrent` emits no register hint, so this is structurally CUDA-only.
2. **32-bit indexing.** The whole domain (479×303×31 ≈ 4.5 M elements) fits
   int32 with a ~470× margin, so the faithful port's `size_t` address math is
   pure overhead. Narrowing `IDX2/IDX3` to `int` frees address registers, which
   lets the 128-register cap land with **less spill** than the size_t
   `epbl_tuned` variant already in the tree (156 vs 276 spill bytes) — the cap
   and the narrower indexing reinforce each other.

## Agreement

`ab_main.cu` builds the state the same way the native driver does (bathymetry →
zstar stretch → thermocline → 2-gyre wind → SST restoring) and runs both
kernels on it. Because the two differ only in launch config and address width —
never in the double arithmetic — kd_int comes out **bit-identical**
(`max|diff| = 0`, field-relative 0, 0/4.5 M elements differ in bits) and MLD is
**bit-identical across all 145 137 columns**.

> **Branch-sensitivity note.** EPBL is branch-chaotic: a sub-ulp FMA difference
> can move the MLD root-find by a whole quantised layer. That did **not** fire
> here (the arithmetic is identical, so there is no sub-ulp difference to
> amplify), so MLD agrees to the bit. Had the two kernels contracted FMAs
> differently, kd_int (the integrated diffusivity) is the robust field-relative
> cross-check and MLD jumps would be expected, not a bug. The harness reports
> both.

## Reproduce

```bash
# build (CPU-only)
nvcc ab_main.cu opt_kernel.cu epbl_kernel.cu -O3 -arch=sm_70 -I../common -I. \
     -DMODEL_NZ_STACK_MAX=128 -DEPBL_LB_THREADS=128 -DEPBL_LB_BLOCKS_PER_SM=4 -o ab

# run THROUGH THE GPU LOCK (shared V100), from the epbl/ dir (needs the bathymetry):
bash ../tmp_local_artifacts/gpu_run.sh epbl \
     bash -c 'cd '"$PWD"' && ./ab 473 297 30 20'      # correctness + timing

# confirm the occupancy lift:
bash ../tmp_local_artifacts/gpu_run.sh epbl bash -c 'cd '"$PWD"' && \
     ncu --launch-count 2 -k "regex:epbl_faithful|epbl_opt" \
       --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers ./ab 473 297 30 1 20'
```

`args`: `nx_phys ny_phys nz reps max_its`. `epbl_kernel.cu` (the faithful
codegen baseline) is untouched; `opt_kernel.cu` is the standalone optimized
artifact, and the EPBL knobs come from the same `-DEPBL_LB_*` the Makefile sets.
