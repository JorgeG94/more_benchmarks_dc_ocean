# CLAUDE.md

Orientation for this repo. Keep it lightweight — details live in `NOTES_ON_PERF.md`
(the answer + method) and each kernel's `README.md`.

## What this is

Minimal reproducers that answer: **does Fortran `do concurrent` cost a
GPU-native ocean model anything vs hand-written CUDA C, and is a C++/HIP rewrite
worth it?** Each kernel is benchmarked as `do concurrent`, a faithful CUDA C
port, and (some) a native `cudaMalloc` driver.

**Answer (`NOTES_ON_PERF.md`):** the OpenACC→CUDA bridge is free, a full rewrite buys
single-digit % of wall time, portable-Fortran fixes buy ~3× more; on the biggest
kernel (`ocean_redi`, ~40%) `do concurrent` already wins. Optimizing a kernel's
*algorithm* (e.g. continuity 11 loops → 3) beats hand-CUDA and ports to DC.

## What you need

A **Fortran compiler** — the compute is standard F2018 `do concurrent`. GPU
offload needs nvfortran (`-stdpar=gpu`); the CPU build (`DATA=none`) works with
any F2018-capable compiler (nvfortran, gfortran ≥ 12, ifx ≥ 2023). `nvcc`/
`hipcc` are optional (only the hand-written GPU-C comparison kernels). Compilers
on `PATH` is enough — no env scripts, no paths outside this repo.

## Build & run

Each active kernel is its own dir; **strategy is a build MODE**, not a dir:

```bash
cd redi                       # any of: continuity_layered redi kappa_shear
                              #   ale_remap btstep epbl meke hll_fluxes hvisc
make dc                       # do concurrent, OpenACC on the GPU        [default]
make dc DATA=omp              # do concurrent via OpenMP target
make dc DATA=none             # do concurrent on the CPU — NO CUDA
make cpp                      # native CUDA (make cpp BACKEND=hip for HIP, structural)
make verify                   # OpenACC dumps a ref, the CPU build cross-checks it
```

Roll-ups from the repo root (over all kernels):

```bash
make all-cuda / all-omp / all-cpu / all-hip / verify-all
make FC=gfortran run-cpu      # build + RUN every kernel on the CPU, small args
make clean                    # everything, whole tree
```

`config.mk` is the one file to edit per machine (compiler, arch, `BACKEND`,
per-compiler flags). Override on the CLI too (`make ARCH=cc90 NVARCH=sm_90 …`).

## Layout

- `<kernel>/` — **active** build-mode kernels (list above). `common/` holds the
  shared macro headers: `directives.h` (Fortran data layer acc/omp/none) +
  `gpu_rt.h` (C++ CUDA↔HIP redirect).
- `legacy_testing/*_benchmark/` — the **original** single-binary DC-vs-CUDA
  benchmarks, retired (kept building until deleted).
- `final_picture/` — collates the legacy benchmarks into the weighted
  C++-vs-Fortran verdict.
- `daxpy_benchmark/`, `continuity_ppm_benchmark/` — old odds-and-ends (no twin).
- `NOTES_ON_PERF.md` — the performance findings + compiler notes; read this first.

## Conventions & gotchas (the non-obvious stuff)

- **Anonymise.** Kernels are extracts of a real model — strip its name and any
  account paths (`rki`/`rakali`/`/home/...`/`/scratch/...`); use `<model>`,
  `<system>` placeholders, as the committed code does.
- **Correctness = bit-identity.** DC and CUDA must agree; the bar is `max rel
  diff < 1e-12` (FMA-contraction level). A perturbation-of-1-ulp check should
  trip the comparison — verify the verifier.
- **Cross-language libm trap:** don't recompute `sin/exp` inputs in the C++
  driver — nvfortran's libm and glibc's differ ~1 ulp. Hand the transcendental
  inputs over via the ref dump and *adopt* them, or you measure libm.
- **`-gpu=tripcount:host` is load-bearing** (NVHPC 26.5 regression, TPR #38714) —
  without it timings are ~2× wrong. It's in the kernel Makefiles; keep it.
- **Two known nvfortran bugs** (reported, since addressed): lost auto-collapse
  when a callee's explicit-shape array is bounded by an integer passed **by
  reference** — fix is to pass the *bounds* by `value` (not the loop indices);
  and lost CSE when a callee does its own array indexing — fix is to pass hoisted
  *scalars*. Inlining the loop body dodges both.
- **`config.mk`:** never put an inline `# comment` on a `VAR ?= value` line — the
  trailing spaces leak into the value and break `-gpu=$(ARCH),mem:separate`.
- **Shared GPU** (V100 analysis node): benchmarks report min-of-reps with warmup;
  for A/B use idle-gating / alternating order, or you measure contention.
- **gfortran here is 8.5** (pre-F2018 `do concurrent local`) — the CPU/omp paths
  are wired via `MODFLAG` but need gfortran ≥ 12 / ifx to actually compile.
