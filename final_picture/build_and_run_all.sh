#!/usr/bin/env bash
# Build (optional), run every benchmark's Fortran bench + native CUDA driver
# serially on an idle GPU, parse each into results/<key>.csv, then collate.
#
#   ./final_picture/build_and_run_all.sh            # run everything, then collate
#   BUILD=1 ./final_picture/build_and_run_all.sh    # `make all-dc` first
#   PARSE_ONLY=1 ./final_picture/build_and_run_all.sh   # re-parse existing logs only
#
# Serial + idle-gated on purpose: the LOGBOOK documents 15-40x swings when the
# V100 is shared. One binary at a time, GPU checked idle before each.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$HERE/.." && pwd)}"
LOG="$HERE/logs"; RES="$HERE/results"
mkdir -p "$LOG" "$RES"
PARSE_ONLY="${PARSE_ONLY:-0}"
BUILD="${BUILD:-0}"
DC_ONLY="${DC_ONLY:-0}"   # run only the do-concurrent side (no CUDA native, no collate)

# lines that isolate the do-concurrent variant + its correctness check
DC_FILTER='do concurrent|^[[:space:]]*DC[ (=]|EPBL do|precomp|flat \+ plain|agreement|bit-ident'

idle_gate () {
  local tries=0 u
  while : ; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ "${u:-100}" -lt 15 ] 2>/dev/null && break
    tries=$((tries+1)); [ $tries -ge 60 ] && { echo "  (gpu still ${u:-?}% after 120s -- proceeding)"; break; }
    sleep 2
  done
}

run_one () {  # key dir bench_cmd native_cmd
  local key="$1" dir="$2" bench="$3" native="$4"
  if [ "$DC_ONLY" = "1" ]; then
    idle_gate; echo "==== $key  ::  $bench"
    ( cd "$ROOT/$dir" && eval "$bench" ) 2>&1 | grep -iE "$DC_FILTER" || true
    echo
    return
  fi
  if [ "$PARSE_ONLY" = "0" ]; then
    idle_gate; echo "==> $key  bench   : $bench"
    ( cd "$ROOT/$dir" && eval "$bench" )  > "$LOG/${key}_bench.txt"  2>&1
    idle_gate; echo "==> $key  native  : $native"
    ( cd "$ROOT/$dir" && eval "$native" ) > "$LOG/${key}_native.txt" 2>&1
  fi
  python3 "$HERE/parse_one.py" "$key" "$LOG/${key}_bench.txt" "$LOG/${key}_native.txt" "$RES/${key}.csv"
}

[ "$BUILD" = "1" ] && { echo "== make all-dc =="; make -C "$ROOT" all-dc || exit 1; }

#        key      dir                            bench command                                              native command
run_one  redi     legacy_testing/redi_benchmark                "./redi_bench 473 297 30 20 10"                            "./redi_native 473 297 30 20 10"
run_one  ks       legacy_testing/kappa_shear_benchmark         "./ks_bench 473 297 30 20 10 1"                            "./ks_native 473 297 30 20 10 1"
run_one  layered  legacy_testing/continuity_layered_benchmark  "./layered_bench 473 297 30 200 10 2"                      "./layered_native 473 297 30 200 10 2"
run_one  ale      legacy_testing/ale_remap_benchmark           "./ale_bench 473 297 30 20 10 25"                          "./ale_native 473 297 30 20 10 25"
run_one  btstep   legacy_testing/btstep_benchmark              "./btstep_bench 473 297 24 200 10"                         "./btstep_native 473 297 24 200 10"
run_one  epbl     legacy_testing/epbl_benchmark                "./epbl_bench 473 297 30 20 8 1 0 20 5"                    "./epbl_native 473 297 30 20 8 1 0 20 5"
run_one  meke     legacy_testing/meke_benchmark                "./meke_bench"                                             "./meke_native"
run_one  flux     legacy_testing/hll_fluxes_benchmark          "FLUX_NPHYS_X=473 FLUX_NPHYS_Y=297 FLUX_REPS=2000 ./flux_bench"  "./flux_native 473 297 2 2000"

[ "$DC_ONLY" = "1" ] && exit 0
echo
python3 "$HERE/collate.py"
