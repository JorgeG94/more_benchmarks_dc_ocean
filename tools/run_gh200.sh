#!/usr/bin/env bash
# tools/run_gh200.sh -- the whole kappa_shear experiment on one GH200 node.
#
#   ./tools/run_gh200.sh              # everything
#   ./tools/run_gh200.sh gpu cmp      # only those phases
#   ./tools/run_gh200.sh -n           # print the plan, run nothing
#
# Phases: gpu cpu threads cmp prod
# Results land in paper_data/ as ks_min_<host>_<stamp>.csv, one file per lane.
# Aggregate with:  tools/ks_collect.py > paper_data/tidy.csv
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# MACHINE FACTS -- stated, never probed. nvidia-smi costs seconds per call on
# Grace-Hopper, which is the whole reason ks_min.sh exists.
# ---------------------------------------------------------------------------
export FC=${FC:-nvfortran}
export ARCH=${ARCH:-cc90}          # H100/GH200
export NVARCH=${NVARCH:-sm_90}
GPU_DEV="GH200"                    # the Hopper die
CPU_DEV="Grace"                    # the 72-core Neoverse V2 socket
CPU_THREADS="1 8 18 36 72"

# The CUDA head-to-head needs an nvcc that still targets this arch. NVHPC's
# bundled CUDA 13.x has begun dropping older ones; if `make cmp` dies with
# "Unsupported gpu architecture", point NVCC at a toolkit that has it, e.g.
#   export NVCC=$NVHPC_ROOT/cuda/12.9/bin/nvcc
export NVCC=${NVCC:-nvcc}

# Production grid (473x297 = 145,137 columns with the halo) and the depth sweep.
# Do NOT shrink the grid: small ones hide launch amortisation on the GPU.
NZ="10 25 30 50 75 100"
PROD_NX=473; PROD_NY=297
CPU_NX=64;   CPU_NY=64             # the grid every other machine's CPU lanes used

export NZ_LIST="$NZ" STACK_POLICIES="prod fit"
export NRUN=${NRUN:-3} KS_TARGET_MS=${KS_TARGET_MS:-1000} COUNTERS=${COUNTERS:-1}

DRY=""; [ "${1:-}" = "-n" ] && { DRY="-n"; shift; }
PHASES="${*:-gpu cpu threads cmp prod}"
run() { echo; echo "### $1"; shift; env "$@" ./tools/ks_min.sh $DRY; }
want() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "=== GH200 end-to-end: $PHASES ==="
command -v "$FC" >/dev/null || { echo "no $FC on PATH"; exit 1; }

# 1. the GPU depth sweep, do concurrent via OpenACC
want gpu && run "GPU / do concurrent (OpenACC)" \
   MODE=dc_gpu DATA=acc DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY

# 2. the same source through the OpenMP-target layer -- the vendor-portable path
want gpu && run "GPU / do concurrent (OpenMP target)" \
   MODE=dc_gpu DATA=omp DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY

# 3. CPU: is `do concurrent` free? serial_do is the baseline, dc_serial the test.
want cpu && run "Grace / serial do loops" \
   MODE=serial_do DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY GPU_ARCH_LABEL=none
want cpu && run "Grace / do concurrent, serial" \
   MODE=dc_serial DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY GPU_ARCH_LABEL=none

# 4. thread scaling, and the PRODUCTION-GRID point.
#    At 64^2 the 3D fields are L3-resident on Grace and at 473x297 they are not
#    -- worth 10-18% at 72 threads. Any GPU-vs-CPU ratio built from the small
#    grid alone is biased in the CPU's favour by that much.
want threads && run "Grace / thread scaling (small grid)" \
   MODE=dc_multicore DEVICE="$CPU_DEV" NXP=$CPU_NX NYP=$CPU_NY \
   THREAD_LIST="$CPU_THREADS" GPU_ARCH_LABEL=none
want threads && run "Grace / production grid, full socket" \
   MODE=dc_multicore DEVICE="$CPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   THREAD_LIST="1 72" STACK_POLICIES=prod GPU_ARCH_LABEL=none

# 5. the head-to-head: dc, cuda_faithful and cuda_opt from ONE binary on the
#    SAME device arrays. FULLCOPY is derived from BCOPY so the two sides copy
#    the same amount -- the pairing that makes the ratio mean anything.
want cmp && run "GPU / CUDA head-to-head" \
   MODE=dc_gpu DATA=acc CUDA=on DEVICE="$GPU_DEV" NXP=$PROD_NX NYP=$PROD_NY

# 6. production physics. NOT a tuning knob -- DT_THERM 7200 (not 300), KD 1.5e-5
#    (not 1e-7), MAX_RINO_IT 25, Wright EOS. Costs 5-7x the wall time because the
#    work per column genuinely goes up, and moves dc/cuda by +5-6% AGAINST DC.
want prod && run "GPU / CUDA head-to-head, PRODUCTION physics" \
   MODE=dc_gpu DATA=acc CUDA=on PHYS=prod DEVICE="$GPU_DEV" \
   NXP=$PROD_NX NYP=$PROD_NY
want prod && run "Grace / production grid + physics" \
   MODE=dc_multicore PHYS=prod DEVICE="$CPU_DEV" NXP=$PROD_NX NYP=$PROD_NY \
   THREAD_LIST="1 72" STACK_POLICIES=prod GPU_ARCH_LABEL=none

echo; echo "=== done. CSVs: ==="
ls -1t paper_data/ks_min_*.csv | head -12
