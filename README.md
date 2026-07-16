# `do concurrent` vs hand-written CUDA/HIP — GPU-native ocean-model kernel benchmarks

Does Fortran `do concurrent` cost this GPU-native ocean model anything versus
hand-written CUDA C — and is a C++/HIP rewrite worth it? This repo isolates the
model's hottest kernels as minimal reproducers and measures each one three ways:
`do concurrent`, a faithful CUDA C port, and a native `cudaMalloc` driver.

**The answer is in [`NOTES_ON_PERF.md`](NOTES_ON_PERF.md):** the OpenACC→CUDA bridge is free
(native ties `host_data`-launched CUDA in every kernel), a full rewrite buys
single-digit % of wall time, portable-Fortran fixes buy ~3× more, and on the
biggest kernel (`ocean_redi`, ~40% of runtime) `do concurrent` already wins.

## What you need

- **A Fortran compiler.** The `do concurrent` compute is standard Fortran 2018 —
  nothing in it is vendor-specific. The portable CPU build (`DATA=none`) works
  with any compiler that supports F2018 `do concurrent … local(…)`: **nvfortran,
  gfortran ≥ 12, or ifx ≥ 2023**. GPU offload needs nvfortran (`-stdpar=gpu`), or
  amdflang / ifx for their own GPUs.
- **Optional: a CUDA or HIP compiler** (`nvcc` / `hipcc`) — only for the
  hand-written GPU-C comparison kernels. Leave it out and you still get the full
  do-concurrent story on the CPU or GPU.
- Nothing else. No environment scripts, no `source`, no paths outside this repo —
  just put the compiler(s) on your `PATH`.

## Compiler flags

Set the compiler once in [`config.mk`](config.mk) (`FC ?= nvfortran`), or override
on the command line (`make FC=gfortran …`). The base flags per compiler:

| compiler   | `do concurrent` runs on | base flags (`FFLAGS_BASE`) |
|------------|-------------------------|----------------------------|
| nvfortran  | NVIDIA GPU or CPU       | `-Mfree -Mbackslash -O3 -fast` |
| gfortran   | CPU                     | `-O3` |
| ifx        | CPU (or Intel GPU)      | `-O3` |
| amdflang   | AMD GPU or CPU          | `-O3` |

The GPU **data layer** is added on top by the build mode and is compiler-specific:

| mode | flags added | who |
|---|---|---|
| `DATA=acc`  | `-stdpar=gpu -acc=gpu -gpu=<arch>,mem:separate` | nvfortran (the measured path) |
| `DATA=omp`  | `-stdpar=gpu -mp=gpu` | nvfortran now; the OpenMP-target route AMD/Intel use |
| `DATA=none` | `-stdpar=multicore` (nvfortran) / nothing (others) | the portable CPU build |

Arch: `ARCH=cc70` (V100) / `cc80` (A100) / `cc90` (H200); for the C kernels
`NVARCH=sm_70/80/90`. All settable in `config.mk` or on the CLI.

## Build & run

Each kernel is its own dir; the strategy is a **build mode**, not a directory:

```bash
cd redi
make dc                    # do concurrent, OpenACC on the GPU            (default)
make dc DATA=none          # do concurrent on the CPU — no CUDA, any Fortran compiler
make dc DATA=omp           # do concurrent via OpenMP target
make cpp                   # the hand-written CUDA C comparison           (needs nvcc)
make cpp BACKEND=hip       # ...or HIP                                    (needs hipcc)
make verify                # cross-check the modes bit-identically
```

Or drive them all from the top:

```bash
make all-cuda              # every kernel: do concurrent (OpenACC) + native CUDA
make all-omp               # every kernel via OpenMP target
make all-cpu               # every kernel on the CPU
make verify-all            # cross-check every kernel (OpenACC vs CPU, bit-identical)
make FC=gfortran all-cpu   # ...with gfortran (≥ 12) instead of nvfortran
make clean                 # remove all build + run artifacts
```

## Where things are

- [`NOTES_ON_PERF.md`](NOTES_ON_PERF.md) — the answer, the method, the profile
  that ranks the kernels, and the compiler findings. Read this first.
- `common/` — `directives.h` (Fortran data-layer macros) and `gpu_rt.h` (the C++
  CUDA↔HIP redirect): the single-sourced plumbing every kernel shares.
- `<kernel>/` — the build-mode kernels (`redi`, `kappa_shear`,
  `continuity_layered`, `ale_remap`, `btstep`, `epbl`, `meke`, `hll_fluxes`,
  `hvisc`), each with its own README.
- `legacy_testing/*_benchmark/` — the original single-binary DC-vs-CUDA
  benchmarks, retired (kept building until deleted).
- `final_picture/` — collates the legacy benchmarks into the weighted
  C++-vs-Fortran verdict.
- `nvbug_*/` — the two filed nvfortran bug reproducers.
- `daxpy_benchmark/`, `continuity_ppm_benchmark/` — old odds-and-ends (no twin).

## What's been validated where

The numbers were measured with **nvfortran (NVHPC 26.5) + nvcc (CUDA 12.9) on a
V100**, which is what the build defaults to. `DATA=omp` is bit-identical to
`DATA=acc` on nvfortran, and `DATA=none` runs on the CPU. gfortran/ifx CPU builds
and amdflang/HIP GPU builds are wired through `config.mk` but need those
toolchains present to exercise — each kernel's README notes what ran where.
