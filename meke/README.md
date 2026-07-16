# meke — build-mode layout (ported from `../meke_benchmark`)

The MEKE step (eddy-kinetic-energy parameterisation): **many small 2-D loops**
on a ~145k-cell domain — 16 `do concurrent` loops per step at the one config
that turns MEKE on (`gabight_sph_meke_v100.nml`). Ported to the new layout:
**one kernel dir, single-sourced compute, strategy chosen as a build MODE**
(not a directory), with the plumbing macro'd. Mirrors the `../continuity_layered`
pilot. The old `../meke_benchmark/` stays as the results-of-record.

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `../common/directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `../common/gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

Compute is single-sourced: `meke.F90` (the production `do concurrent` kernels,
verbatim from `ocean_meke.F90`) and `meke_kernel.cu` (the hand-written C++
kernels) never change between modes — only the macro lines do. The one directive
edit in `meke.F90` is `!$acc routine seq` → `DC_ROUTINE_SEQ` on `meke_inv_lmix`.

## Build / run

```bash
make dc                 # do concurrent, OpenACC on the NVIDIA GPU   [default]
make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA now; AMD/Intel with their compilers)
make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA, NO OpenACC
make cpp                # native C++/CUDA (cudaMalloc + CUDA graph), no Fortran
make cpp BACKEND=hip    # native C++/HIP -- structural (needs ROCm/hipcc)
make verify             # OpenACC-GPU dumps a ref; the CPU build recomputes + cross-checks it
make run-dc / run-cpp / clean
```

`-gpu=tripcount:host` is **mandatory** on the GPU paths (NVIDIA TPR #38714) — the
many-loop MEKE region otherwise refreshes trip counts from the device copy per
kernel (~2x slower). Copied verbatim from production's flags.

## Measured on this node (V100, shared with 6 sibling agents — timing NOISY), 473×297×30, 20 reps

| mode | ms/step | cross-check vs OpenACC ref |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | ~0.39 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | ~0.37 | **max\|diff\| = 0.0** (bit-identical) |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~6–28 (noisy) | **max rel = 2.2e-16** (FMA-level) |
| `cpp BACKEND=cuda` faithful (16 kernels) | ~0.24 | final `meke` range matches DC |
| `cpp BACKEND=cuda` fused (6 kernels) | ~0.20 | — |
| `cpp BACKEND=cuda` graph (6 kern, 1 graph) | ~0.19 | — |

**What this proves:** the macro'd data layer (`directives.h`) is a numerically
inert swap on nvfortran across OpenACC / OpenMP / host, and the do-concurrent
MEKE step builds and runs on the **CPU with zero CUDA**. The native driver's
degenerate-config invariants (`le==0`, `kh_diff==0`, `bottom_fac2==1`,
`barotr_fac2==1`) all hold, so the port exercises MEKE's SHAPE faithfully.

**What it does NOT yet prove:** true AMD/Intel portability needs *their*
compilers (`amdflang` / `ifx`) — `DATA=omp` here still runs through nvfortran;
`BACKEND=hip` needs a ROCm toolchain to compile. Both are toolchain availability,
not design gaps.

## Layout

```
meke/
  constants.F90 meke_state.F90   shared stubs (verbatim from ../meke_benchmark)
  meke.F90                       production do-concurrent kernels (routine-seq -> DC_ROUTINE_SEQ)
  meke_kernel.cu  meke_args.h    hand-written C++ kernels + shared ABI/index macros
  drivers/
    dc_main.F90                  DC-only driver (data via macros; dump/cross-check)
    cpp_main.cu                  native cudaMalloc + CUDA-graph driver
  Makefile                       build modes: dc DATA=... / cpp BACKEND=...
```

The production variants `meke_dt` / `meke_fused` / `meke_fused_acc` / `meke_acc`
from the old dir are deliberately **not** carried over — this is the DC-vs-native
port only.
