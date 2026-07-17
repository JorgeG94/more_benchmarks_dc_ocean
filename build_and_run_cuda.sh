#!/usr/bin/env bash
# Build + run every kernel's hand-written GPU-C driver. Default nvcc/CUDA;
# pass `hip` for the AMD/ROCm path (hipcc). NOTE: the HIP path is structural
# on this repo's dev node (no ROCm) -- it builds the switch but must be run on
# an AMD box; see config.mk's BACKEND=hip block.
#
#   ./build_and_run_cuda.sh [cuda|hip]      (default cuda)
#   size comes from benchmark_config.mk
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

backend="${1:-cuda}"
[ "$backend" = cuda ] || [ "$backend" = hip ] || { echo "usage: $0 [cuda|hip]"; exit 2; }
banner "hand-written GPU-C  (BACKEND=$backend)"
ensure_lock; gpu_assert_free || exit 1
csv="$DATADIR/gpuc_${backend}_$(stamp).csv"
csv_init "$csv" "kernel,backend,nx,ny,nz,ms_per_step"

printf '  %-22s %14s\n' kernel ms/step
for k in "${ALL_KERNELS[@]}"; do
   ms=$(build_run "$k" "cpp BACKEND=$backend" "cpp_$backend" "$k-$backend" 1)
   printf '  %-22s %14s\n' "$k" "${ms:-**FAIL**}"
   csv_row "$csv" "$k,$backend,$NXP,$NYP,$NZ,${ms:-NaN}"
done
echo "  -> $csv"
