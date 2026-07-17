# CLAUDE.md

Orientation for this repo. Keep it lightweight — details live in `NOTES_ON_PERF.md`
(the answer + method) and each kernel's `README.md`.

## What this is

Minimal reproducers that answer: **does Fortran `do concurrent` cost a
GPU-native ocean model anything vs hand-written CUDA C, and is a C++/HIP rewrite
worth it?** Each kernel is benchmarked as `do concurrent`, a faithful CUDA C
port, and (some) a native `cudaMalloc` driver.

**Answer (`NOTES_ON_PERF.md`):** the OpenACC→CUDA bridge is free; both sides
fully optimized, a hand-CUDA rewrite of the ocean core is **~8% faster than
`do concurrent`, measured end-to-end** (whole-model RK2 loop, `ideal_benchmark/`,
V100). Amdahl runs it: the two column solvers (`ocean_redi` + `kappa_shear`) are
~80% of runtime and near-parity, so CUDA's real wins on the cheap kernels can't
move the total. Portable-Fortran optimizations (fusion/hoist) buy 1.18×–1.34×
per kernel, are bit-identical, and — unlike the CUDA-only tricks — port to the CPU.

**Framing this as a paper on _performance portability_:** the thesis is that ONE
`do concurrent` source runs across NVIDIA GPU (`-stdpar=gpu -acc`), AMD/Intel GPU
(`-mp=gpu` OpenMP target), and CPU multicore (`-stdpar=multicore`, any F2018
compiler) at a cost of single-digit % vs a GPU-only hand-CUDA rewrite. See
`NOTES_ON_PERF.md` → "The three measurement layers" and the reproduction guide
below.

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

## Reproducing the study (cold-start guide for the paper)

Three measurement layers, cheapest → most complete. Numbers + traps live in
`NOTES_ON_PERF.md`; this is the *how to run it* list.

- **Per-kernel portability check — `make dc DATA=none` + `make verify`.** Proves
  the same `do concurrent` source runs on the CPU and stays bit-identical to the
  GPU. The portability axis for the paper: sweep `DATA=acc|omp|none` and
  `FC=nvfortran|gfortran|ifx`.
- **Portable-opt A/B — `make dc_ab` (per kernel).** Faithful DC vs optimized DC
  in one binary; times the speedup, checks bit-identity. Runs on GPU *and* CPU
  (`DATA=none`) — same source. (Kernels with an opt: redi, hvisc, ale_remap,
  btstep, meke, continuity_layered.)
- **opt-CUDA vs opt-DC — `make cmp` (per kernel).** Both launched from ONE binary
  on the SAME device arrays via `host_data use_device`. The apples-to-apples
  per-kernel verdict. Mixed nvfortran+nvcc link.
- **Whole-model — `cd ideal_benchmark && make run`.** "Dumb" RK2 (2-stage) loop
  over all 8 ocean kernels, hot GPU, all-DC then all-CUDA, ms/stage + per-kernel
  breakdown. The headline number.

**GPU discipline (load-bearing for correct numbers):**
- The whole-model effect is ~8% ≈ contention noise. **Run on an EXCLUSIVE GPU** —
  confirm `nvidia-smi --query-compute-apps=pid,process_name --format=csv` is
  empty (a co-running MOM6 once flipped the RK2 sign). Then run 4–5× and check
  the ratio is stable to <0.2%.
- Serialize/idle-gate GPU runs through `tmp_local_artifacts/gpu_run.sh <label>
  <cmd>` (flock + wait-for-idle). It is gitignored, recreate if missing.
- Always production size (`473 297 30`); small grids hide the launch-amortization
  gap. Keep `-gpu=tripcount:host` (see gotchas).

**Extending to a CPU whole-model number (a paper figure waiting to be made):**
`ideal_benchmark/` is currently GPU-only (the CUDA pass needs a device). The DC
pass is pure `do concurrent`, so a `DATA=none` build of just the DC side gives an
all-CPU whole-model ms/stage across nvfortran/gfortran/ifx — the natural
CPU-portability half of the story. Not yet wired; the `rk2_<k>_stage` (DC)
entries already exist, the `_stage_cuda` entries just get `#ifdef`'d out.

## Conventions & gotchas (the non-obvious stuff)

- **Correctness = bit-identity.** DC and CUDA must agree; the bar is `max rel
  diff < 1e-12` (FMA-contraction level). A perturbation-of-1-ulp check should
  trip the comparison — verify the verifier.
- **Cross-language libm trap:** don't recompute `sin/exp` inputs in the C++
  driver — nvfortran's libm and glibc's differ ~1 ulp. Hand the transcendental
  inputs over via the ref dump and *adopt* them, or you measure libm.
- **`-gpu=tripcount:host` is load-bearing** (NVHPC 26.5 regression, TPR #38714) —
  without it timings are ~2× wrong. It's in the kernel Makefiles; keep it.
- **`config.mk`:** never put an inline `# comment` on a `VAR ?= value` line — the
  trailing spaces leak into the value and break `-gpu=$(ARCH),mem:separate`.
