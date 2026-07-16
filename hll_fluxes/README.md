# hll_fluxes — build-mode layout

The production **2-D HLL flux kernel** (`kernel_flux.F90`, byte-identical to the
ocean model's shipped file) in the new layout: **one kernel dir, single-sourced
compute, strategy chosen as a build MODE** (not a directory), with the plumbing
macro'd. Mirrors the `../continuity_layered` pilot. The old
`../hll_fluxes_benchmark/` stays as the results-of-record.

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `../common/directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `../common/gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

The compute is single-sourced: `kernel_flux.F90` (the `do concurrent` HLL kernel,
production `compute_flux_hll`) and `flux_kernel.cu` (the hand-written C++ kernel)
never change between modes — only the macro lines do. This kernel carries **no**
`!$acc routine seq` directives: the cross-module `pure` helpers called from inside
`do concurrent` are device-compiled automatically by `-stdpar=gpu`, so there is
nothing here for `directives.h` to macro. The `#include` is kept for consistency.

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

`dc` args are `nx_phys ny_phys reps warm` (2-D, `nghost=2` baked in). The native
C++ driver keeps its own CLI, `nx_phys ny_phys nghost reps`, so `run-cpp` uses a
separate `CPP_ARGS` default.

## Measured on this node (V100, nvfortran 26.5 / nvcc 12.9), 473×297 interior

| mode | ms/rep | cross-check vs OpenACC ref |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | ~0.10 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | ~0.10 | **max\|diff\| = 0.0** (bit-identical) |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~2.3 | **max\|diff\| = 0.0** (bit-identical) |
| `cpp BACKEND=cuda` (native cudaMalloc) | ~0.06 (min) / ~0.12 (mean) | `sum(flux_h)` matches the DC drivers |

Timing is noisy (V100 shared with sibling agents); the cross-check, not the ms,
is the load-bearing result. `sum(flux_h)` agrees to all printed digits between
the native C++ driver and the Fortran DC drivers, so the two kernels compute the
same problem.

**What this proves:** the macro'd data layer (`directives.h`) is a numerically
inert swap on nvfortran across OpenACC / OpenMP / host — the `do concurrent` HLL
kernel builds and runs bit-identically on all three, and on the **CPU with zero
CUDA**. The C++ runtime redirect (`gpu_rt.h`) keeps `flux_kernel.cu` +
`cpp_main.cu` building under nvcc, with hipcc wired behind `-DUSE_HIP`.

**What it does NOT yet prove:** true AMD/Intel portability needs *their* compilers
(`amdflang` / `ifx`) — `DATA=omp` here still runs through nvfortran. `BACKEND=hip`
needs a ROCm toolchain to compile. Both are toolchain availability, not design
gaps. The native C++ driver's bit-exact `verify` path expects a `flux_ref_*.bin`
from the old `flux_bench`, which this layout does not build; without it the driver
reports interior checksums instead (that comparison lived in the old dir and was
left there deliberately).

## Layout

```
hll_fluxes/
  constants.F90 grid.F90  shared stubs (byte-identical to the old dir)
  kernel_flux.F90         the do-concurrent HLL kernel (+ #include "directives.h")
  flux_kernel.cu          the hand-written C++ kernel (cuda_runtime.h -> gpu_rt.h)
  drivers/
    dc_main.F90           DC-only driver (data via macros; dump/cross-check)
    cpp_main.cu           native cudaMalloc driver (was flux_native.cu)
  Makefile                build modes: dc DATA=... / cpp BACKEND=...
```
