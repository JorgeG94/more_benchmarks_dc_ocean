# Notes on performance

The findings, distilled. Numbers are V100 (cc70), NVHPC 26.5 / CUDA 12.9,
production size 473×297×30 unless noted.

## The answer

**A fully hand-optimized CUDA rewrite of the ocean core is ~8% faster than
optimized `do concurrent` — measured end-to-end, not estimated.** Every kernel
is either a memory-bound stencil or a register-bound column solver, and
nvfortran's codegen ties nvcc's closely on both; what gap remains is single-digit
and concentrated in the cheap kernels.

- **Measured, not weighted.** The whole 8-kernel ocean core, run in one hot-GPU
  RK2 loop (`ideal_benchmark/`), is **all-DC 101.5 vs all-CUDA 94.2 ms/stage →
  CUDA +7.8%** (V100, 473×297×30; stable across 5 clean runs). This *replaces*
  the old profile-share estimate with a direct measurement.
- **Amdahl runs the show.** The two column solvers `ocean_redi` + `ocean_vmix_kshear`
  are ~80% of a stage and near-parity (redi favours DC +7%, kappa favours CUDA
  +16%, net CUDA +4%); the remaining ~20% of light kernels is CUDA +27%. Blended
  → +8%. Speeding up a cheap kernel by 1.7× cannot move a total the giants own.
- **The OpenACC→CUDA bridge is free.** A native `cudaMalloc` driver ties the
  `host_data use_device` path in every kernel (~0.004%–10 µs/call), so the
  single-binary head-to-heads below are apples-to-apples: one device image, one
  clock, opt-CUDA vs opt-DC.
- **The algorithm is where the money is, and it's language-neutral** — the same
  portable fusion/hoist helps CUDA and DC, and (unlike the CUDA-only tricks)
  ports straight to the CPU. Portable Fortran optimizations landed 1.18×–1.34×
  over each kernel's own DC baseline, all bit-identical, all CPU-portable.

> Earlier drafts claimed "DC beats CUDA" on `btstep` and "DC ties on the giants."
> Both were **ratio-of-ratios / contention artifacts** — comparing a standalone
> nvfortran binary's speedup against a standalone nvcc binary's, and (in one case)
> a co-running job inflating the second-timed pass. The single-binary `cmp`
> harness and the exclusive-GPU RK2 runs corrected them. Keep the caveat in mind.

## The single-binary head-to-head (`make cmp` per kernel)

`opt-CUDA` and `opt-DC` launched from ONE binary on the SAME device arrays via
`!$acc host_data use_device` — the allocator/harness variable removed. Agreement
is FMA-contraction level (≤1e-14) everywhere, so these are true codegen deltas.

| kernel | opt-DC ms | opt-CUDA ms | winner |
|---|---:|---:|---|
| ocean_redi | 40.0 | 42.8 | **DC +7%** |
| ocean_hvisc | 1.36 | 1.42 | **DC +4%** |
| ocean_ale_remap | 8.4 | 7.0 | CUDA +20% |
| ocean_barotropic (btstep) | 2.87 | 2.67 | CUDA +7% (graph +13%) |
| MEKE | 0.30 | 0.19 | CUDA +57% |

DC wins the memory-bound stencils where nvfortran's codegen is as good or better;
CUDA wins the launch-count-bound light kernels (its async/graph edge is real but
low-leverage). `ocean_vmix_kshear` (register/divergence-bound, no opt on either
side) is CUDA +16% in the RK2 breakdown — the one heavy kernel that favours CUDA.

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
- **The whole-model number needs an EXCLUSIVE GPU.** The end-to-end effect is
  ~8% — the *same size as contention noise* from a co-running job (e.g. MOM6).
  A single contended run flipped the RK2 sign (CUDA +8% → DC +8%) because the
  second-timed (CUDA) pass caught the other job. `tmp_local_artifacts/gpu_run.sh`
  idle-gates (<12% util) + serializes with `flock`, but that is NOT enough
  against a *running* job — it can fire in a between-timestep lull. Confirm
  `nvidia-smi --query-compute-apps` is empty, then run 4–5× and check the ratio
  is stable to <0.2% (a clean run is: per-kernel isolated sum ≈ the aggregate).
- **Trust the single-binary harness over ratio-of-ratios.** Comparing an
  nvfortran binary's speedup to a separate nvcc binary's speedup is not a
  language comparison — allocator, warmup, and clock differ. Put both on one
  device image (`make cmp`) before claiming a winner.
- **Measure at production sizes.** The dc/cuda gap is a strong function of total
  cell count (launch amortization) — 4096² makes everything look clean while
  nothing runs there.

## The three measurement layers (what to run)

1. **`make dc_ab`** (per kernel) — faithful DC vs optimized DC in one binary;
   proves the portable optimization is bit-identical and times the speedup. Runs
   on GPU (`DATA=acc`) *and* CPU (`DATA=none`) — same source, so it doubles as
   the CPU-portability check.
2. **`make cmp`** (per kernel) — optimized DC vs optimized CUDA on the SAME
   device arrays via `host_data use_device`. The apples-to-apples per-kernel
   verdict (table above). Mixed nvfortran+nvcc link; `-cuda` is link-only (the
   bridge module must NOT see it → NVFORTRAN-S-0528).
3. **`ideal_benchmark/`** — a "dumb" RK2 (2-stage) driver that runs all 8 ocean
   kernels back-to-back per step on a hot GPU, all-DC then all-CUDA, and reports
   ms/stage + the per-kernel breakdown. This is the model-level answer. Isolation
   trick: each kernel (they redefine `constants`/`grid`/… with different bodies)
   is compiled into its own `-Bsymbolic` `.so` so duplicate module symbols never
   cross-bind; only the `bind(C)` `rk2_<k>_*` entries are global. `hll_fluxes` is
   excluded (coastal SWE, not ocean core); `epbl` stays.
