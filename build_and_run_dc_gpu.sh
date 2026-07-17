#!/usr/bin/env bash
# Build + run every kernel as `do concurrent` on the GPU. Default is the
# measured path (OpenACC data layer, DATA=acc); pass `omp` for the OpenMP-
# target layer (the route AMD/Intel GPUs take). Same source as the CPU build.
#
#   ./build_and_run_dc_gpu.sh [acc|omp]     (default acc)
#   size comes from benchmark_config.mk
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

layer="${1:-acc}"
[ "$layer" = acc ] || [ "$layer" = omp ] || { echo "usage: $0 [acc|omp]"; exit 2; }
banner "do concurrent on the GPU  (DATA=$layer)"
ensure_lock; gpu_assert_free || exit 1
csv="$DATADIR/dc_gpu_${layer}_$(stamp).csv"
csv_init "$csv" "kernel,layer,nx,ny,nz,ms_per_step"

printf '  %-22s %14s\n' kernel ms/step
for k in "${ALL_KERNELS[@]}"; do
   ms=$(build_run "$k" "dc DATA=$layer" "dc_$layer" "$k-dc$layer" 1)
   printf '  %-22s %14s\n' "$k" "${ms:-**FAIL**}"
   csv_row "$csv" "$k,$layer,$NXP,$NYP,$NZ,${ms:-NaN}"
done
echo "  -> $csv"
