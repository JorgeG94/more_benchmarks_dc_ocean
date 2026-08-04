#!/usr/bin/env bash
# tools/run_aurora.sh -- the whole kappa_shear experiment on an Aurora node.
#
#   ./tools/run_aurora.sh             # everything
#   ./tools/run_aurora.sh cpu         # one phase
#   ./tools/run_aurora.sh -n          # print the plan, run nothing
#
# Phases: gpu cpu threads prod
# do concurrent ONLY -- there is no hand-written SYCL twin, by design.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# MACHINE FACTS
#   Nothing to load -- ifx is on PATH by default. (2025.3 and 2026.0 both work.)
# ifx exposed a real conformance bug in this code once: dc_main.F90 declared
# locals named `grid` and `ks` while USE-ing modules of those names. nvfortran
# and gfortran accept it, ifx rejects with #6450. Fixed -- noted so a rebuild
# failure there is recognised rather than re-diagnosed.
# ---------------------------------------------------------------------------
export FC=${FC:-ifx}
GPU_DEV="Intel GPU"                # ONE TILE -- that is what a rank gets
CPU_DEV="Xeon Max"                 # 104-core Sapphire Rapids + HBM
CPU_THREADS="1 26 52 104"

# The OpenMP-target lane. Same source; only these flags differ, and they REPLACE
# the Makefile's DC_MODE_FLAGS wholesale.
export DC_GPU_FLAGS="-qopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent -DDC_DATA_OMP"
export GPU_ARCH_LABEL="spir64"

NZ="10 25 30 50 75 100"
PROD_NX=473; PROD_NY=297
CPU_NX=64;   CPU_NY=64

export NZ_LIST="$NZ" CUDA=off
export NRUN=${NRUN:-3} KS_TARGET_MS=${KS_TARGET_MS:-1000} COUNTERS=${COUNTERS:-1}

DRY=""; [ "${1:-}" = "-n" ] && { DRY="-n"; shift; }
PHASES="${*:-gpu cpu threads prod}"
run() { echo; echo "### $1"; shift; env "$@" ./tools/ks_min.sh $DRY; }
want() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "=== Aurora end-to-end ($FC): $PHASES ==="
command -v "$FC" >/dev/null || { echo "no $FC on PATH (expected by default on Aurora)"; exit 1; }

# 1. the GPU depth sweep, both frame settings. The frame axis is cheap to carry
#    and it is the one place AMD and NVIDIA disagree violently, so measure it
#    here too rather than assuming Intel behaves like either.
want gpu && run "Intel Max (1 tile) / do concurrent" \
   MODE=dc_gpu DATA=omp DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   STACK_POLICIES="prod fit"

# 2. CPU: is `do concurrent` free? ifx has been the cleanest of the four
#    compilers on this (0.996-1.005), so a deviation here is a signal.
want cpu && run "Xeon Max / serial do loops" \
   MODE=serial_do DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY \
   STACK_POLICIES="prod fit" GPU_ARCH_LABEL=none DC_GPU_FLAGS=
want cpu && run "Xeon Max / do concurrent, serial" \
   MODE=dc_serial DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY \
   STACK_POLICIES="prod fit" GPU_ARCH_LABEL=none DC_GPU_FLAGS=

# 3. thread scaling. ifx maps do concurrent onto threads under -qopenmp
#    (config.mk supplies it) and reads OMP_NUM_THREADS; ks_min.sh sets that and
#    ACC_NUM_CORES together, because nvfortran reads the other one and a curve
#    flat at 1.00x is what you get if only one is set.
want threads && run "Xeon Max / thread scaling" \
   MODE=dc_multicore DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY \
   THREAD_LIST="$CPU_THREADS" STACK_POLICIES=prod GPU_ARCH_LABEL=none DC_GPU_FLAGS=

# 4. production physics + the production-grid CPU point (removes the
#    cache-residency confound in any GPU-vs-socket ratio).
want prod && run "Intel Max / PRODUCTION physics" \
   MODE=dc_gpu DATA=omp PHYS=prod DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   STACK_POLICIES="prod fit"
want prod && run "Xeon Max / production grid, full socket" \
   MODE=dc_multicore DEVICE="$CPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   THREAD_LIST="1 104" STACK_POLICIES=prod GPU_ARCH_LABEL=none DC_GPU_FLAGS=

echo; echo "=== done. CSVs: ==="
ls -1t paper_data/ks_min_*.csv | head -12
