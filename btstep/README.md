# btstep — build-mode layout (barotropic substep, inner substep loop)

The 2-D closed-basin barotropic solver, ported to the new layout: **one kernel
dir, single-sourced compute, strategy chosen as a build MODE** (not a directory),
with the plumbing macro'd. Mirrors `../continuity_layered` (the pilot) and
`../dc_patterns` (config.mk + a directives header). The old
`../btstep_benchmark/` stays as the results-of-record.

Each timed call runs `n_inner = 24` substeps (7 live `do concurrent` loops each)
wrapped in production's `!$acc kernels async(1)` / `!$acc wait(1)` idiom.

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

Compute is single-sourced: `btstep.F90` (the `do concurrent` substep) and
`btstep_kernel.cu` (the hand-written 11-/5-kernel + CUDA-graph port) never change
between modes — only the macro lines do.

## Build / run

```bash
make dc                 # do concurrent, OpenACC on the NVIDIA GPU   [default]
make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA now; AMD/Intel with their compilers)
make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA, NO OpenACC
make cpp                # native C++/CUDA (cudaMalloc + CUDA graphs), no Fortran
make cpp BACKEND=hip    # native C++/HIP -- structural (needs ROCm/hipcc)
make verify             # OpenACC-GPU dumps a ref; CPU + OpenMP recompute + cross-check it
make run-dc / run-cpp / clean
```

`dc_main.F90` is DC-only: NO CUDA, NO `host_data`, NO nvcc. The CUDA-comparison
states (`wc/wp_/wf`) of the old bench are dropped — a single working set `w`.

## Measured on this node (V100, nvfortran 26.5 / nvcc 12.9), 473×297, n_inner=24

Shared V100 (6 sibling agents), so ms/call is noisy — reported, not a gate.

| mode | ms/call | cross-check vs OpenACC ref (`max\|d eta\|`) |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | ~3.7 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | ~7.8 | **0.0** (bit-identical) |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~300 | **7.8e-16** → field-rel 3.2e-15 |
| `cpp BACKEND=cuda` faithful / fused / graph | 3.14 / 2.69 / 2.51 | internal fusion max\|d eta\| = 0.0 |

Cross-check bar is `max|d eta|` judged against the **field magnitude** `max|eta|`
(3.2e-15 ≪ 1e-12), exactly as `../btstep_benchmark/btstep_bench.F90` scores it —
bt_eta is divergence-driven and has interior cells ~0, so pointwise relative
error is the wrong bar there.

**What this proves:** the macro'd data layer (`directives.h`) is a numerically
inert swap on nvfortran across OpenACC / OpenMP / host, and the do-concurrent
substep builds and runs on the **CPU with zero CUDA**.

**Note on the omp timing.** `DATA=omp` is ~2× slower than `acc` here, and that is
expected, not a port defect: btstep's compute directive is the production
`!$acc kernels async(1)` idiom, which is honoured only under `-acc`. Under
`-mp=gpu` (no `-acc`) those lines are inert comments, so each `do concurrent`
loop runs as a **blocking** `-stdpar=gpu` launch and pays the per-loop host sync
the async queue removes on the `acc` path. The numbers are identical (bit-for-bit
above); only the launch overlap is lost. A true async-OpenMP variant would need an
`!$omp ... nowait` macro, which `directives.h` does not (yet) carry — see below.

**What it does NOT yet prove:** true AMD/Intel portability needs *their*
compilers (`amdflang`/`ifx`) — `DATA=omp` here still runs through nvfortran. And
`BACKEND=hip` needs a ROCm toolchain. Both are toolchain availability, not design
gaps.

## Layout

```
btstep/
  constants.F90 bt_state.F90   shared stubs (grid/metrics/coriolis/work types)
  btstep.F90                   the do-concurrent substep (production variant)
  btstep_kernel.cu             the hand-written C++ kernels (faithful/fused/graph)
  btstep_args.h                shared C ABI (BtArgs struct + launch prototype)
  drivers/
    dc_main.F90                DC-only driver (data via macros; dump/cross-check)
    cpp_main.cu                native cudaMalloc + main() driver (no OpenACC)
  Makefile                     build modes: dc DATA=... / cpp BACKEND=...
```

## Deviations from the pilot (`../continuity_layered`)

- **No `!$acc routine seq` to macro.** btstep's kernel has none; instead it
  carries `!$acc kernels async(1)` / `!$acc end kernels` / `!$acc wait(1)`
  compute directives, left **bare** (inert without `-acc`). `directives.h` has no
  macro for them, and none is needed for correctness — see the omp note above.
  The only edit to `btstep.F90` is the added `#include "directives.h"`.
- **Cross-check bar is field-relative** (`max|d eta| / max|eta|`), matching the
  original benchmark, not the pilot's pointwise-relative bar (which would flag
  ~0 interior cells). The pilot's CPU path came out bit-identical; here the
  GPU→CPU FMA/reduction reorder gives ~1e-15, correctly judged inert.
- `-I.` added to `CUFLAGS` so `drivers/cpp_main.cu` finds `btstep_args.h` in the
  dir root.
