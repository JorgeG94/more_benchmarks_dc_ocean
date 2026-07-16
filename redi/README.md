# redi — build-mode layout (Redi neutral diffusion)

The Redi neutral-diffusion kernel ported to the new layout: **one kernel dir,
single-sourced compute, strategy chosen as a build MODE** (not a directory),
with the plumbing macro'd. Mirrors the `../continuity_layered/` pilot. The old
`../redi_benchmark/` stays as the results-of-record.

The production do-concurrent path is the pair `redi_calc_coeffs` (Phase A,
per-face neutral-surface sweep) + `redi_apply_flux` (Phase B, per-tracer
along-neutral flux divergence), verbatim from `ocean_redi.F90`. One timed rep =
both phases for 2 tracers (temperature + salinity).

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `../common/directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `../common/gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

The compute is single-sourced: `ocean_redi.F90` / `ocean_eos.F90` (the
`do concurrent` kernels, `!$acc routine seq` -> `DC_ROUTINE_SEQ`) and
`redi_kernel.cu` (the C++ kernel) never change between modes — only the macro
lines do. The DC driver maps the whole working set itself with
`DC_ENTER_IN`/`DC_ENTER_CREATE`, so the module's vestigial
`enter_data`/`exit_data` type-bound procedures are left verbatim and simply
never called.

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

`DEFS = -DMODEL_NZ_STACK_MAX=128 -DWITH_CUDA` is appended to BOTH the Fortran
and the CUDA compile flags. `NZ_STACK_MAX=128` is production's value (not the
256 header default); `nz=30` needs `2*nz+2 = 62 <= 128`. `-gpu=tripcount:host`
is MANDATORY for Redi on NVHPC 26.5 (TPR #38714) and is carried on both GPU
modes.

## Measured on this node (V100, nvfortran 26.5 / nvcc 12.9), 473×297×30, 8 reps / 2 warm

| mode | ms/rep | cross-check vs OpenACC ref |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | ~45 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | ~45 | **max\|diff\| = 0.0** (bit-identical) |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~1400 | **max\|diff\| = 4.5e-13, max rel 2.1e-16** (FMA-level) |
| `cpp BACKEND=cuda` (native cudaMalloc) | ~45 | runs standalone (self-computed IC) |

Timing is NOISY: the V100 is shared with sibling agents, so a given run drifts
(acc has been seen at 45–70 ms/rep across runs). The cross-check figures are
contention-proof and are the point of this port.

**What this proves:** the macro'd data layer (`directives.h`) is a numerically
inert swap on nvfortran across OpenACC / OpenMP / host for the Redi kernel —
including the nested-allocatable state (`ms%tracers(:)%hTr`, an array of derived
type with an allocatable component), whose deep copy the OpenMP-target path
handles here bit-for-bit. The do-concurrent kernel builds and runs on the CPU
with zero CUDA.

**What it does NOT prove:** true AMD/Intel portability needs *their* compilers
(`amdflang` / `ifx`) — `DATA=omp` here still runs through nvfortran; and
`BACKEND=hip` needs a ROCm toolchain. Both are toolchain availability, not
design gaps.

## Layout

```
redi/
  constants.F90 grid.F90 ocean_metrics.F90 multilayer_cgrid_state.F90
  ocean_boundary_types.F90                shared MRE stubs
  ocean_eos.F90                           EOS point routines (routine-seq -> DC_ROUTINE_SEQ)
  ocean_redi.F90                          the do-concurrent Redi kernel (routine-seq -> DC_ROUTINE_SEQ)
  redi_kernel.cu                          the hand-written C++ kernel (cuda_runtime.h -> gpu_rt.h)
  drivers/
    dc_main.F90                           DC-only driver (data via macros; dump/cross-check)
    cpp_main.cu                           native cudaMalloc/hipMalloc driver
  Makefile                                build modes: dc DATA=... / cpp BACKEND=...
```

Dropped from the old `../redi_benchmark/`: `redi_cuda.F90` (OpenACC->CUDA
bridge), `redi_probe.cu` (host_data micro-probe), `ocean_redi_pre.F90` (the
PRECOMP optimisation) — none is needed for a DC-only + native-C++ port.
