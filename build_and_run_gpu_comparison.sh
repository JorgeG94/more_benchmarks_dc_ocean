#!/usr/bin/env bash
# Per-kernel DC vs hand-GPU-C on the GPU: builds and times both for every
# kernel and tabulates the ratio. Default compares against CUDA; pass `hip`
# for AMD.
#
# These are SEPARATE binaries (dc_acc vs cpp_<backend>) -- the honest
# per-kernel table, but with the harness/allocator caveat. The rigorous
# same-device-image number (opt-DC vs opt-CUDA in one binary via host_data)
# is `make cmp` per kernel; see NOTES_ON_PERF.md.
#
#   ./build_and_run_gpu_comparison.sh [cuda|hip]     (default cuda)
#   size comes from benchmark_config.mk
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

backend="${1:-cuda}"
[ "$backend" = cuda ] || [ "$backend" = hip ] || { echo "usage: $0 [cuda|hip]"; exit 2; }
banner "DC vs $backend  (per kernel; separate binaries -- see 'make cmp' for same-image)"
ensure_lock; gpu_assert_free || exit 1
csv="$DATADIR/gpu_comparison_${backend}_$(stamp).csv"
csv_init "$csv" "kernel,dc_ms,gpuc_ms,dc_over_gpuc,backend,nx,ny,nz"

printf '  %-22s %11s %11s %10s\n' kernel "DC ms" "$backend ms" "DC/$backend"
for k in "${ALL_KERNELS[@]}"; do
   dc=$(build_run "$k" "dc DATA=acc"            "dc_acc"        "$k-cmp-dc" 1)
   gc=$(build_run "$k" "cpp BACKEND=$backend"   "cpp_$backend"  "$k-cmp-gc" 1)
   ratio=$(awk -v a="${dc:-0}" -v b="${gc:-0}" 'BEGIN{ if(b>0) printf "%.3f", a/b; else print "NaN" }')
   printf '  %-22s %11s %11s %10s\n' "$k" "${dc:-FAIL}" "${gc:-FAIL}" "$ratio"
   csv_row "$csv" "$k,${dc:-NaN},${gc:-NaN},$ratio,$backend,$NXP,$NYP,$NZ"
done
echo "  -> $csv"
echo "  (DC/$backend > 1 means $backend is faster; < 1 means do concurrent is faster)"
