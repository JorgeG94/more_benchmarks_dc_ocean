# Notes on performance

The findings, distilled. Numbers are V100 (cc70), NVHPC 26.5 / CUDA 12.9,
production size 473×297×30 unless noted.

## The answer

**`do concurrent` is not costing this ocean model anything meaningful vs
hand-written CUDA C.** Every kernel measured is either a memory-bound stencil or
a register-bound column solver, and nvfortran's codegen ties nvcc's on both — the
gap is a flat launch-overhead residual that vanishes at scale.

- **The OpenACC→CUDA bridge is free.** A native `cudaMalloc` driver ties the
  `host_data use_device` path in every kernel (~0.004%–10 µs/call). So the CUDA
  numbers were never flattering to Fortran.
- **A full C++/HIP rewrite buys single-digit % of total wall time.** Portable
  Fortran fixes (algorithm + signatures + fusion) buy ~3× more, and on the
  biggest kernel (`ocean_redi`, ~40%) `do concurrent` already wins.
- **The algorithm is where the money is, and it's language-neutral** — the same
  change helps CUDA and DC.

## Per-kernel: DC vs faithful CUDA (bit-identical unless noted)

Profile shares are the model's (biggest first).

| kernel | share | dc / cuda (faithful) | note |
|---|---:|---|---|
| ocean_redi | ~40% | Fortran **wins** | precompute hoist (8→5 PPM columns) = 1.35×, portable |
| ocean_vmix_kshear | ~18% | ~1.05× | iterative Picard solver; register-bound |
| ocean_continuity | ~7% | ~1.11× | fuses to 1.5× (CUDA) / 1.36× (DC) — see below |
| ocean_ale_remap | ~6% | ~1.06× | sigfix + fusion = 1.45× portable |
| ocean_hvisc (Smagorinsky) | ~5% | **0.98×** (DC edges) | full closure: strain→A_h + Laplacian apply |
| ocean_barotropic (btstep) | ~5% | fused DC **beats** fused CUDA | only CUDA graphs pull ahead (~6%) |
| ocean_vmix_epbl | ~2% | ~1.05× codegen | the 1.33× that exists is a **compiler bug** (below) |
| MEKE | ~1% | — | async + fusion = 1.67× portable |
| coastal HLL flux | — | flat DC **beats** CUDA | after fixing the lost-collapse bug |

## Case study: optimizing continuity_layered

11 PPM kernels → 3 by fusing each direction's reconstruction + boundary +
transport (recompute the upwind cell in registers, so the `hfl/hfr` scratch
arrays never touch global memory: ~280 MB/call gone) + 32-bit indexing.

- **CUDA: 0.93 → 0.62 ms = 1.50×.** **DC: 1.11 → 0.82 ms = 1.36×.** Bit-identical.
- **Fused DC (0.82) beats the *original* faithful CUDA (0.93).** Optimized CUDA
  then reopens a ~1.32× gap (the fused recon is register-heavy, where nvfortran
  trails nvcc — same story as kappa_shear / the HLL flat body).
- **Everything cleverer LOST** (kept as `OPTVER 3-8`): full recompute-fusion,
  k-loop / k-blocking for ILP, `__launch_bounds__`, div+flux fusion, shared-
  memory tiling. The kernel is memory-bound; the flat one-thread-per-face layout
  lets the GPU's own L2 + occupancy do the reuse better than any hand-rolled
  scheme. **Only removing genuine waste paid off** (the intermediate arrays,
  64-bit addressing). `kO_div` is already ~84% of DRAM peak.

## Compiler findings

Diagnose the regime first: launch-bound (→ fuse/async), compute-bound (→ less
arithmetic/precision), spill-bound (→ less redundant per-column work),
divergence-bound (→ algorithmic).

- **`-gpu=tripcount:host` is load-bearing** (NVHPC 26.5 regression, NVIDIA
  TPR #38714): without it, device-side trip-counts insert a per-kernel data
  refresh in multi-loop regions → timings ~2× wrong. It's in the Makefiles.
- **Bug 1 — lost auto-collapse:** a `do concurrent` stops
  collapsing when a callee's explicit-shape array dummy is bounded by an integer
  passed **by reference** + the body has a call. **Fix: pass the array *bounds*
  by `value`** (`nx,ny` — verified necessary & sufficient; the loop indices
  `i,j` by value do nothing).
- **Bug 2 — lost CSE:** redundant global loads when an
  inlined callee does its own array indexing. **Fix: hoist reads and pass
  scalars.** Inlining the loop body dodges both bug 1 and bug 2.
- **Bug 3 — `maxregcount` quality:** at an identical register cap nvfortran
  spills far more than nvcc's `__launch_bounds__` (EPBL: 568 B → 10.95 ms vs
  160 B → 4.87 ms). Config-dependent — it *helped* kappa_shear. Not universal.

## Methodology / traps

- **Correctness = bit-identity**, bar `max rel diff < 1e-12` (FMA-contraction
  level). Verify the verifier: perturb a term by 1 ulp and confirm it trips.
- **Cross-language libm trap:** don't recompute `sin/exp` inputs in the C++
  driver — nvfortran's libm and glibc's differ ~1 ulp, amplified by cancellation.
  Hand the transcendental inputs over via the ref dump and *adopt* them, or the
  cross-check measures libm, not the kernel. Use field-relative error for fields
  with near-zero cells.
- **Shared V100:** report min-of-reps with warmup; for A/B, idle-gate and
  alternate order, or you measure contention.
- **Measure at production sizes.** The dc/cuda gap is a strong function of total
  cell count (launch amortization) — 4096² makes everything look clean while
  nothing runs there.
