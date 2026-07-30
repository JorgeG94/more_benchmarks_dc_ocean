#!/usr/bin/env bash
# Build + run every kernel as the SERIAL DO-LOOP baseline (plain nested `do`,
# committed <kernel>/serialdo/ sources -- NOT do concurrent). The paper's
# "serial_do" column. No -stdpar/-acc/-mp, no GPU, no CUDA; portable to any
# F2008+ compiler INCLUDING gfortran (which cannot compile the do-concurrent
# source at all).
#
#   ./build_and_run_serial_do.sh [FC] [FCFLAGS]
#     FC       compiler (default gfortran). e.g. nvfortran | ifx | flang | gfortran
#     FCFLAGS  override the base compile flags (default: config.mk's FFLAGS_BASE)
#
#   Problem size comes from benchmark_config.mk. Writes the unified provenance
#   CSV (see benchmark_common.sh: CSV_COLS) so it stacks with every other mode.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

FC="${1:-${FC:-gfortran}}"
FCFLAGS="${2:-${FCFLAGS:-}}"
mkargs=(FC="$FC")
[ -n "$FCFLAGS" ] && mkargs+=(FFLAGS_BASE="$FCFLAGS")

command -v "$FC" >/dev/null 2>&1 || { echo "ERROR: compiler '$FC' not found on PATH." >&2; exit 1; }
ver="$(fc_version "$FC")"
flags="$(cd "$REPO/redi" && make -s print-SD_FFLAGS "${mkargs[@]}" 2>/dev/null)"
dev="$(device_for serial_do)"

banner "SERIAL DO-LOOP baseline  (FC=$FC)"
echo "  compiler : $ver"
echo "  flags    : $flags"
echo "  device   : $dev"
echo "  git      : $GHASH    host: $HOSTN"
echo

csv="$DATADIR/serial_do_${FC##*/}_${TS}.csv"; csv_new "$csv"
printf '  %-22s %-12s %14s\n' kernel build ms/step
printf -- '  ---------------------------------------------------\n'
for k in "${ALL_KERNELS[@]}"; do
   rm -rf "$REPO/$k/build/serialdo"
   if out=$( cd "$REPO/$k" && make serialdo "${mkargs[@]}" 2>&1 ); then
      run=$( cd "$REPO/$k" && OMP_NUM_THREADS=1 ./sdo $ARGS 2>&1 )
      ms=$(parse_ms "$run")
      printf '  %-22s %-12s %14s\n' "$k" "built" "${ms:-**FAIL**}"
      emit_row "$csv" "$k" serial_do "$FC" "$ver" "$flags" 1 "$dev" "${ms:-NaN}"
   else
      ( cd "$REPO/$k" && echo "$out" > .serialdo_build.log )
      printf '  %-22s %-12s %s\n' "$k" "FAILED" "see $k/.serialdo_build.log"
      emit_row "$csv" "$k" serial_do "$FC" "$ver" "$flags" 1 "$dev" NaN
   fi
done
echo; echo "  -> $csv"
