# continuity_layered — build-mode pilot

Proof-of-concept for the new layout: **one kernel dir, single-sourced compute,
strategy chosen as a build MODE** (not a directory), with the plumbing macro'd.
Mirrors `../dc_patterns` (config.mk + a directives header). The old
`../continuity_layered_benchmark/` stays as the results-of-record until every
kernel is ported.

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

The compute is single-sourced: `continuity_layered.F90` (the `do concurrent`
kernel) and `layered_kernel.cu` (the C++ kernel) never change between modes —
only the macro lines do.

## Build / run

```bash
make dc                 # do concurrent, OpenACC on the NVIDIA GPU   [default]
make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA now; AMD/Intel with their compilers)
make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA, NO OpenACC
make cpp                # native C++/CUDA (cudaMalloc), no Fortran
make cpp BACKEND=hip    # native C++/HIP -- structural (needs ROCm/hipcc)
make verify            # OpenACC-GPU dumps a ref; the CPU build recomputes + cross-checks it
make run-dc / run-cpp / clean
```

## Measured on this node (V100, nvfortran 26.5 / nvcc 12.9), 473×297×30

| mode | ms/rep | cross-check vs OpenACC ref |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | 1.10 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | 1.11 | **max\|diff\| = 0.0** (bit-identical) |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~22 | **max\|diff\| = 0.0** (bit-identical) |
| `cpp BACKEND=cuda` (native cudaMalloc) | 0.97 | matches the old `layered_native` |

**What this proves:** the macro'd data layer (`directives.h`) is a numerically
inert swap on nvfortran across OpenACC / OpenMP / host, and the do-concurrent
kernel builds and runs on the **CPU with zero CUDA**. The C++ runtime redirect
(`gpu_rt.h`) keeps one `.cu` building under nvcc, with hipcc wired behind
`-DUSE_HIP`.

**What it does NOT yet prove:** true AMD/Intel portability needs *their*
compilers (`amdflang` / `ifx`) — `DATA=omp` here still runs through nvfortran.
And `BACKEND=hip` needs a ROCm toolchain to compile. Both are toolchain
availability, not design gaps: the seams exist and are exercised as far as this
node allows.

## Layout

The two macro headers are shared across all ported kernels in `../common/`
(`directives.h`, `gpu_rt.h`) so they cannot drift; each kernel's Makefile
adds `-I../common`.

```
continuity_layered/
  constants.F90 grid.F90  shared stubs
  continuity_layered.F90  the do-concurrent kernel (routine-seq -> DC_ROUTINE_SEQ)
  layered_kernel.cu       the hand-written C++ kernel
  drivers/
    dc_main.F90           DC-only driver (data via macros; dump/cross-check)
    cpp_main.cu           native cudaMalloc/hipMalloc driver
  Makefile                build modes: dc DATA=... / cpp BACKEND=...
```
