#!/usr/bin/env bash
# tools/run_frontier.sh -- the whole kappa_shear experiment on a Frontier node.
#
#   ./tools/run_frontier.sh           # everything
#   ./tools/run_frontier.sh gpu       # one phase
#   ./tools/run_frontier.sh -n        # print the plan, run nothing
#
# Phases: gpu frame cpu threads prod
# do concurrent ONLY -- there is no hand-written HIP twin, by design.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# MACHINE FACTS
#   module load PrgEnv-amd    (or: module load rocm; the compiler is amdflang)
# The compiler is amdflang, NOT flang -- config.mk knows every name LLVM flang
# ships under, but the offload flags below must name the arch explicitly:
# rocminfo on a login node may report nothing and gfx90a would be assumed anyway.
# ---------------------------------------------------------------------------
export FC=${FC:-amdflang}
AMD_ARCH=${AMD_ARCH:-gfx90a}       # MI250X. MI300 is gfx942.
GPU_DEV="MI250X"                   # ONE GCD -- that is what a rank gets
CPU_DEV="EPYC 7A53"
CPU_THREADS="1 8 16 32 56"

# The OpenMP-target lane. Same `do concurrent` source as everywhere else; only
# these flags differ. They REPLACE the Makefile's DC_MODE_FLAGS wholesale.
export DC_GPU_FLAGS="-fopenmp --offload-arch=$AMD_ARCH -fdo-concurrent-to-openmp=device -DDC_DATA_OMP"
export GPU_ARCH_LABEL="$AMD_ARCH"

NZ="10 25 30 50 75 100"
PROD_NX=473; PROD_NY=297
CPU_NX=64;   CPU_NY=64

export NZ_LIST="$NZ" CUDA=off
export NRUN=${NRUN:-3} KS_TARGET_MS=${KS_TARGET_MS:-1000} COUNTERS=${COUNTERS:-1}

DRY=""; [ "${1:-}" = "-n" ] && { DRY="-n"; shift; }
PHASES="${*:-gpu frame cpu threads prod}"
run() { echo; echo "### $1"; shift; env "$@" ./tools/ks_min.sh $DRY; }
want() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "=== Frontier end-to-end ($FC, $AMD_ARCH): $PHASES ==="
command -v "$FC" >/dev/null || { echo "no $FC on PATH -- module load PrgEnv-amd"; exit 1; }

# 1. the GPU depth sweep at both frame settings.
#    NZ_STACK_MAX IS THE WHOLE BALLGAME HERE and the NVIDIA-derived "the frame
#    doesn't matter" conclusion does NOT transfer: prod/fit costs 14.8x at nz=10
#    and ~3x at nz=25-30, against 0.98-1.13x on V100/GH200. Never collapse this
#    axis on AMD.
want gpu && run "MI250X (1 GCD) / do concurrent" \
   MODE=dc_gpu DATA=omp DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   STACK_POLICIES="prod fit"

# 2. where the frame penalty sets in -- sweep NZ_STACK_MAX at FIXED depth, which
#    is the only way to separate "more layers of work" from "a fixed frame".
want frame && run "MI250X / frame sweep at nz=30" \
   MODE=dc_gpu DATA=omp DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   NZ_LIST=30 STACK_POLICIES="31 40 48 64 96 128"

# 3. CPU: is `do concurrent` free on this compiler? flang-family is the one
#    that is NOT free at the shallow end (1.125x at nz=10, gone by nz=50), so
#    this lane needs the full depth range to state the caveat honestly.
want cpu && run "EPYC 7A53 / serial do loops" \
   MODE=serial_do DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY \
   STACK_POLICIES="prod fit" GPU_ARCH_LABEL=none DC_GPU_FLAGS=
want cpu && run "EPYC 7A53 / do concurrent, serial" \
   MODE=dc_serial DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY \
   STACK_POLICIES="prod fit" GPU_ARCH_LABEL=none DC_GPU_FLAGS=

# 4. thread scaling. flang maps DC to threads with -fdo-concurrent-to-openmp=host
#    (config.mk supplies it); ks_min.sh sets OMP_NUM_THREADS *and* ACC_NUM_CORES
#    per run because different compilers read different ones.
want threads && run "EPYC 7A53 / thread scaling" \
   MODE=dc_multicore DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY \
   THREAD_LIST="$CPU_THREADS" STACK_POLICIES=prod GPU_ARCH_LABEL=none DC_GPU_FLAGS=

# 5. production physics + production grid. The CPU point removes the
#    cache-residency confound: at 64^2 the 3D fields sit in cache, at 473x297
#    they do not, and any GPU-vs-socket ratio from the small grid flatters the CPU.
want prod && run "MI250X / PRODUCTION physics" \
   MODE=dc_gpu DATA=omp PHYS=prod DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   STACK_POLICIES="prod fit"
want prod && run "EPYC 7A53 / production grid, full socket" \
   MODE=dc_multicore DEVICE="$CPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   THREAD_LIST="1 56" STACK_POLICIES=prod GPU_ARCH_LABEL=none DC_GPU_FLAGS=

echo; echo "=== done. CSVs: ==="
ls -1t paper_data/ks_min_*.csv | head -12
