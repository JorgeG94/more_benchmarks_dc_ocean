#!/bin/bash
# Paired A/B: CUDA-via-Fortran (host_data) vs CUDA native (cudaMalloc).
#
# WHY THIS EXISTS: up to ~9 agents share this V100 and contention swings timings
# by more than the effect under test (an 804x rep spread was observed). A single
# run of each driver is worthless here. This:
#   * waits for an IDLE GPU before EACH sample (nvidia-smi compute-apps == 0),
#   * re-checks idle AFTER the sample and discards it if anyone showed up,
#   * runs the two drivers back-to-back so they see the same conditions,
#   * alternates which driver goes first, so a drift in load cannot
#     systematically favour whichever one is always measured first,
#   * reports MIN across samples -- the least-disturbed observation, which is the
#     only contention-robust statistic available. Means here measure the
#     neighbours, not the kernel.
#
# Even so: an idle-gated sample is not an idle machine. Treat every number as
# PROVISIONAL until re-run serially. Bit-identity (flux_native's verify) is
# unaffected by contention and needs none of this.
#
#   ./ab_idle.sh <nphys_x> <nphys_y> <reps> <samples>

set -u
NX=${1:-4096}; NY=${2:-4096}; REPS=${3:-20}; SAMPLES=${4:-7}

busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }

wait_idle() {
    for _ in $(seq 1 90); do
        [ "$(busy)" -eq 0 ] && sleep 1 && [ "$(busy)" -eq 0 ] && return 0
        sleep 2
    done
    return 1
}

echo "=== paired A/B: ${NX}x${NY} interior, ${REPS} reps/driver, ${SAMPLES} samples ==="
printf "%-8s %12s %12s %12s %10s\n" "sample" "fort-CUDA" "nat-mean" "nat-min" "verdict"

fort_list=(); nat_list=(); natmin_list=(); kept=0
for s in $(seq 1 "$SAMPLES"); do
    if ! wait_idle; then printf "%-8s %s\n" "$s" "GPU never went idle - skipped"; continue; fi

    if [ $((s % 2)) -eq 1 ]; then
        f=$(FLUX_NPHYS_X=$NX FLUX_NPHYS_Y=$NY FLUX_REPS=$REPS ./flux_bench 2>/dev/null | grep 'CUDA C ' | sed 's/.*: *//;s/ ms.*//')
        o=$(./flux_native "$NX" "$NY" 2 "$REPS" 2>/dev/null)
    else
        o=$(./flux_native "$NX" "$NY" 2 "$REPS" 2>/dev/null)
        f=$(FLUX_NPHYS_X=$NX FLUX_NPHYS_Y=$NY FLUX_REPS=$REPS ./flux_bench 2>/dev/null | grep 'CUDA C ' | sed 's/.*: *//;s/ ms.*//')
    fi
    n=$(echo "$o"  | grep 'mean  '  | head -1 | sed 's/.*: *//;s/ ms.*//')
    m=$(echo "$o"  | grep 'min of'  | sed 's/.*: *//;s/ ms.*//')

    if [ "$(busy)" -ne 0 ]; then
        printf "%-8s %12s %12s %12s %10s\n" "$s" "$f" "$n" "$m" "DISCARD(busy)"
        continue
    fi
    printf "%-8s %12s %12s %12s %10s\n" "$s" "$f" "$n" "$m" "keep"
    fort_list+=("$f"); nat_list+=("$n"); natmin_list+=("$m"); kept=$((kept+1))
done

[ "$kept" -eq 0 ] && { echo "no clean samples"; exit 1; }

python3 - "$kept" "${fort_list[*]}" "${nat_list[*]}" "${natmin_list[*]}" <<'EOF'
import sys
kept = int(sys.argv[1])
f = [float(x) for x in sys.argv[2].split()]
n = [float(x) for x in sys.argv[3].split()]
m = [float(x) for x in sys.argv[4].split()]
print("-" * 60)
print(f"  clean samples                        : {kept}")
print(f"  CUDA via Fortran driver (host_data)  : {min(f):8.4f} ms   (min of {kept})")
print(f"  CUDA native (cudaMalloc, no OpenACC) : {min(n):8.4f} ms   (min of {kept})")
print(f"  ratio  fortran / native              : {min(f)/min(n):8.4f} x")
print(f"  native per-rep floor (device events) : {min(m):8.4f} ms")
print("-" * 60)
EOF
