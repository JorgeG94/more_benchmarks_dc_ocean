#!/usr/bin/env bash
# Build + run every kernel as `do concurrent` on the GPU. Default is the measured
# NVIDIA path (OpenACC data layer, DATA=acc); pass `omp` for the OpenMP-target
# layer, which is the vendor-portable one (NVIDIA / AMD / Intel). Same source as
# the CPU builds. The paper's "dc_gpu_<layer>" column. Emits the unified
# provenance CSV (see benchmark_common.sh: CSV_COLS).
#
#   ./build_and_run_dc_gpu.sh [acc|omp] [FC]     (default acc, nvfortran)
#   ./build_and_run_dc_gpu.sh omp amdflang       AMD GPU (arch: AMD_GPU_ARCH)
#   ./build_and_run_dc_gpu.sh omp ifx            Intel GPU
#   size comes from benchmark_config.mk
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

layer="${1:-acc}"; FC="${2:-${FC:-nvfortran}}"
[ "$layer" = acc ] || [ "$layer" = omp ] || { echo "usage: $0 [acc|omp] [FC]"; exit 2; }
command -v "$FC" >/dev/null 2>&1 || { echo "ERROR: compiler '$FC' not found on PATH." >&2; exit 1; }

# ---- vendor-aware offload flags -------------------------------------------
# The kernel Makefiles' DATA=acc|omp branches are spelled for nvfortran
# (-stdpar=gpu -acc/-mp=gpu -gpu=<ccXX>). No other vendor understands those, so
# for a non-nvfortran compiler we inject config.mk's DC_GPU_FLAGS as a
# DC_MODE_FLAGS override on a DATA=omp build: the OpenMP-target *data layer* in
# common/directives.h is vendor-portable, only the compile flags differ. This is
# the same override run_all.sh applies -- keep the two in step.
mkflags=()
case "${FC##*/}" in
   nvfortran) ;;                    # native path: the Makefile's own DATA=$layer flags
   *)
      if [ "$layer" != omp ]; then
         echo "ERROR: $FC has no OpenACC offload path -- OpenACC here is nvfortran-only." >&2
         echo "       Use the OpenMP-target layer instead:  $0 omp $FC" >&2
         exit 2
      fi
      gpuf="$(cd "$REPO/redi" && make -s print-DC_GPU_FLAGS FC="$FC")"
      if [ -z "$gpuf" ]; then
         echo "ERROR: config.mk has no DC_GPU_FLAGS for '${FC##*/}' -- add a branch for it" >&2
         echo "       (see the 'vendor GPU-offload flags' block in config.mk)." >&2
         exit 2
      fi
      mkflags=("DC_MODE_FLAGS=$gpuf")
      ;;
esac

mode="dc_gpu_$layer"
ver="$(fc_version "$FC")"
# print WITH the override applied, so the CSV records the flags actually used
flags="$(cd "$REPO/redi" && make -s print-BASE_FFLAGS FC=$FC) $(cd "$REPO/redi" && make -s print-DC_MODE_FLAGS DATA=$layer FC=$FC ${mkflags[@]+"${mkflags[@]}"})"
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
   if out=$( cd "$REPO/$k" && make dc DATA=$layer FC=$FC ${mkflags[@]+"${mkflags[@]}"} 2>&1 ); then
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
