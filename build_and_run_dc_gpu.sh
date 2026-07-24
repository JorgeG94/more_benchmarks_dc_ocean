#!/usr/bin/env bash
# Build + run every kernel as `do concurrent` on the GPU (nvfortran). Default is
# the measured path (OpenACC data layer, DATA=acc); pass `omp` for the OpenMP-
# target layer. Same source as the CPU builds. The paper's "dc_gpu_<layer>"
# column. Emits the unified provenance CSV (see benchmark_common.sh: CSV_COLS).
#
#   ./build_and_run_dc_gpu.sh [acc|omp] [FC]     (default acc, nvfortran)
#   size comes from benchmark_config.mk
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

layer="${1:-acc}"; FC="${2:-${FC:-nvfortran}}"
[ "$layer" = acc ] || [ "$layer" = omp ] || { echo "usage: $0 [acc|omp] [FC]"; exit 2; }
command -v "$FC" >/dev/null 2>&1 || { echo "ERROR: compiler '$FC' not found on PATH." >&2; exit 1; }

mode="dc_gpu_$layer"
ver="$(fc_version "$FC")"
flags="$(cd "$REPO/redi" && make -s print-BASE_FFLAGS FC=$FC) $(cd "$REPO/redi" && make -s print-DC_MODE_FLAGS DATA=$layer FC=$FC)"
dev="$(device_for "$mode")"

banner "do concurrent on the GPU  (DATA=$layer, FC=$FC)"
echo "  compiler : $ver"
echo "  flags    : $flags"
echo "  device   : $dev"
echo "  git      : $GHASH    host: $HOSTN"; echo
ensure_lock; gpu_assert_free || exit 1

csv="$DATADIR/dc_gpu_${layer}_${TS}.csv"; csv_new "$csv"
printf '  %-22s %-12s %14s\n' kernel build ms/step
printf -- '  ---------------------------------------------------\n'
for k in "${ALL_KERNELS[@]}"; do
   rm -rf "$REPO/$k/build/dc_$layer"
   if out=$( cd "$REPO/$k" && make dc DATA=$layer FC=$FC 2>&1 ); then
      run=$( cd "$REPO/$k" && "$LOCK" "$k-$mode" ./dc_$layer $ARGS 2>&1 )
      ms=$(parse_ms "$run")
      printf '  %-22s %-12s %14s\n' "$k" "built" "${ms:-**FAIL**}"
      emit_row "$csv" "$k" "$mode" "$FC" "$ver" "$flags" 1 "$dev" "${ms:-NaN}"
   else
      ( cd "$REPO/$k" && echo "$out" > .dc_gpu_build.log )
      printf '  %-22s %-12s %s\n' "$k" "FAILED" "see $k/.dc_gpu_build.log"
      emit_row "$csv" "$k" "$mode" "$FC" "$ver" "$flags" 1 "$dev" NaN
   fi
done
echo; echo "  -> $csv"
