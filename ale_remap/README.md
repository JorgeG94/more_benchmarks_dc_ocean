# ale_remap — build-mode port

Ports `../ale_remap_benchmark` to the new layout: **one kernel dir,
single-sourced compute, strategy chosen as a build MODE** (not a directory),
with the plumbing macro'd. Mirrors the `../continuity_layered` pilot.

The COMPUTE is the model's production `ocean_apply_ale_remap_step` as shipped —
the do-concurrent orchestrator (`ale_remap.F90`) plus the byte-identical
per-column PPM remap (`kernel_remap.F90`). Only the **as-shipped** DC variant is
carried; the `_acc` / `_fix` / `_fused` experiment variants from the old dir are
dropped.

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `../common/directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `../common/gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

`ale_remap.F90` and `kernel_remap.F90` never change between modes — every
`!$acc routine seq` became `DC_ROUTINE_SEQ`; the `do concurrent` / `local(...)`
compute is untouched. `-DMODEL_NZ_STACK_MAX=128` (production's real build value)
and `-gpu=tripcount:host` (mandatory, NVIDIA TPR #38714) are applied to both
paths.

## Build / run

```bash
make dc                 # do concurrent, OpenACC on the NVIDIA GPU   [default]
make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA now; AMD/Intel with their compilers)
make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA, NO OpenACC
make cpp                # native C++/CUDA (cudaMalloc), no Fortran
make cpp BACKEND=hip    # native C++/HIP -- structural (needs ROCm/hipcc)
make verify             # OpenACC-GPU dumps a ref; the CPU build recomputes + cross-checks it
make run-dc / run-cpp / clean
```

Args: `nx_pts ny_pts nz reps warm hdrift%` (default `473 297 30 200 10 25`). The
h-drift % is the ONE state knob the cost is sensitive to — it sets how many old
layers each new layer overlaps (the PPM sweep's inner trip count). The state is
restored from a pristine copy before every timed rep because the remap has a
fixed point; un-restored reps would measure strictly less work than production.

## Measured on this node (V100-PCIE-32GB, nvfortran / nvcc 12.x), 473×297×30, h-drift 25%

Timing is NOISY — the V100 was shared by several agents during this run, so
treat the ms/rep as order-of-magnitude, not a ranking. The load-bearing result
is the **cross-check**, which is exact.

| mode | ms/rep | cross-check vs OpenACC ref |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | ~11 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | ~11 | **max\|diff\| = 0.0** (bit-identical) |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~1900 | **max\|diff\| = 0.0** (bit-identical) |
| `cpp BACKEND=cuda` (native cudaMalloc) | ~7–8 | runs; matches the old `ale_native` |

(reps 8 / warm 2 used for the quick verify above; the shipped default is 200/10.)

**What this proves:** the macro'd data layer (`directives.h`) is a numerically
inert swap on nvfortran across OpenACC / OpenMP / host — including the
`ms%tracers(t)%hTr` array-of-derived-type deep copy, which the OpenMP-target
path maps and runs correctly here. The do-concurrent remap builds and runs on
the **CPU with zero CUDA**. The C++ runtime redirect (`gpu_rt.h`) keeps one
`.cu` building under nvcc, with hipcc wired behind `-DUSE_HIP`.

**What it does NOT yet prove:** true AMD/Intel portability needs *their*
compilers — `DATA=omp` here still runs through nvfortran. `BACKEND=hip` needs a
ROCm toolchain to compile. Both are toolchain availability, not design gaps.

## Layout

```
ale_remap/
  constants.F90 remap_state.F90   MRE stubs (types + parameters)
  kernel_remap.F90                verbatim per-column PPM remap (routine-seq -> DC_ROUTINE_SEQ)
  ale_remap.F90                   production DC orchestrator, as-shipped variant
  ale_kernel.cu                   the hand-written C++/CUDA kernels
  drivers/
    dc_main.F90                   DC-only driver (data via macros; restore + dump/cross-check)
    cpp_main.cu                   native cudaMalloc/hipMalloc driver (copy of ale_native.cu)
  Makefile                        build modes: dc DATA=... / cpp BACKEND=...
```
