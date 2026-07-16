# Optimizing the coastal HLL barotropic-flux CUDA kernel

A best-CUDA-practices pass over `flux_kernel.cu` (the faithful transliteration of
`kernel_flux.F90`'s HLL path), freed from the "faithful" rule but held
**bit-identical** to the original: the per-cell arithmetic in `opt_kernel.cu` is
copied line-for-line — same order, same parenthesisation, same intermediates —
so `ab_main.cu` measures **max|diff| = 0** on all five outputs. This kernel has
no sqrt-cancellation to launder as a rewrite; only the *addressing* and *launch
shape* change. V100 (sm_70), grid `473×297`, nghost=2.

## Result

**Bit-identical, ~1.01× (i.e. essentially neutral).** The faithful port is
already near-optimal; the only bit-identical change that is ever *non-negative*
is 32-bit indexing, and it buys a marginal win that grows slightly with grid
size (where address math is a bigger share of the work).

| grid | faithful ms/rep | optimized ms/rep | speedup | max\|diff\| |
|---|---|---|---|---|
| 473×297×2  | 0.0605 | 0.0601 | 1.007× | 0 |
| 945×594×2  | 0.2097 | 0.2074 | 1.011× | 0 |
| 1500×1500×2| 0.783  | 0.778  | 1.006× | 0 |

At the target size the win is within run-to-run noise (repeat runs landed
1.000×–1.007×); at larger grids it is small but consistently positive.

## What worked (the default build: `IDX32=1`, 32×8 block)

**32-bit indexing.** The faithful `IDX` casts to `size_t` (64-bit) at every one
of the ~40 stencil references per thread. The domain fits `int32`
(`477×301 = 1.4e5 ≪ 2³¹`), so the 64-bit address math is pure overhead. Narrowing
`IDX` to `int` is bit-identical (the indices are the same integers, only the
address type is narrower) and drops the kernel from **104 → 100 registers**, with
**0 spill** either way. It is a genuine "remove waste" change — just a small one
here, because 104 and 100 registers land at the **same occupancy** (see below).

## Why it's only ~1.01× — the kernel is already at its ceiling

The faithful kernel is a **register-heavy, one-thread-per-cell** flat body: it
computes all four interface HLL fluxes (each with its own velocity extraction,
wave speeds, and slope-limited reconstruction) plus the well-balanced divergence
in registers, writing only the five output arrays — no intermediate global
traffic to eliminate. The "fuse the face flux + divergence, recompute in
registers" playbook that won elsewhere is **already the faithful layout here**;
there is nothing left to fuse.

Occupancy is **register-limited**. On the V100, both 104 and 100 regs/thread
resolve to **2 blocks/SM = 16 warps = 25% occupancy** — the register reduction
from 32-bit indexing doesn't cross an occupancy boundary, which is exactly why it
is neutral rather than a win. Raising occupancy requires cutting registers below
~85 (24 warps) or ~64 (32 warps), and the only way to force that is `ptxas`
pressure, which **spills** — and the spill traffic costs more than the occupancy
buys, on a kernel that already has abundant per-thread ILP to hide latency.

## What was tried and LOST

| idea | mechanism | regs (spill) | result @473×297 | why it lost |
|---|---|---|---|---|
| `__launch_bounds__(256,3)` | force ≤85 regs → 3 blk/SM (37.5%) | 80 (96 B) | **0.905×** | spill traffic > occupancy gain |
| `__launch_bounds__(256,4)` | force ≤64 regs → 4 blk/SM (50%)  | 64 (312 B) | **0.625×** | heavy spill dominates |
| `__launch_bounds__(128,6)` | 128-block, ≤80 regs | 80 (96 B) | 0.868× | same, + smaller blocks |
| block 32×4 (128 thr) | more/smaller blocks | 100 (0) | 0.921× | same 16 warps, worse tails |
| block 64×4 | wider blocks | 100 (0) | 0.976× | no occupancy change |
| block 32×16 (512 thr) | fewer/bigger blocks | 100 (0) | 0.925× | same 16 warps, coarser |

Every occupancy-raising lever lost. The 32×8 default already coalesces along
Fortran's fast (column-major) index — one warp per row of the stencil — and the
kernel's own ILP hides latency better than extra warps bought through spills.
These knobs are kept in `opt_kernel.cu` as `-DLB=`, `-DBX=`, `-DBY=`, `-DIDX32=`
for reproducibility.

## Reproduce

```bash
cd hll_fluxes
nvcc ab_main.cu opt_kernel.cu flux_kernel.cu -O3 -arch=sm_70 -I../common -o ab
# ALL GPU runs go through the shared lock:
bash ../tmp_local_artifacts/gpu_run.sh hll ./ab 473 297 2 2000
# sweep a knob:  nvcc ... -DLB=3   (launch_bounds)   -DBY=4 (block)   -DIDX32=0 (size_t)
```

## Caveats

- `max|diff| = 0` is exact and verified on **all five** outputs (`flux_h`,
  `flux_hu`, `flux_hv`, `mass_flux_x`, `mass_flux_y`) over the interior — the
  region the kernel writes. The optimized kernel runs the *same arithmetic*, so
  this is expected, not luck.
- The state is **all-wet by construction** (both depths ≫ `THIN_LAYER_THRESHOLD`),
  so the dry/thin branches never fire and warps don't diverge on them — same as
  the native driver. A wet/dry coastline is a different experiment and could
  shift the compute/occupancy balance.
- `IDX32=1` is safe only while `nx*ny < 2³¹`; `-DIDX32=0` restores the
  byte-identical `size_t` path for a pathologically large single tile.
- The honest headline: this kernel is a case where the **faithful port is already
  the optimized port**. The measurable, bit-identical improvement is ~1.01×;
  everything cleverer than "narrow the index type" makes it slower.
