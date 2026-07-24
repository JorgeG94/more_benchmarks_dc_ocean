#!/usr/bin/env bash
# Per-kernel DC vs hand-GPU-C on the GPU: builds and times both for every kernel
# and tabulates the ratio. Default compares against CUDA; pass `hip` for AMD.
#
# SEPARATE binaries (dc_acc vs cpp_<backend>) -- the honest per-kernel table but
# with the harness/allocator caveat. The rigorous same-device-image number is
# `make cmp` per kernel; see NOTES_ON_PERF.md.
#
# Writes TWO rows per kernel to the unified provenance CSV -- one dc_gpu_acc, one
# gpuc_<backend> -- so the data stacks with every other mode; the DC/GPU-C ratio
# is printed to the console (and derivable from the two rows downstream).
#
#   ./build_and_run_gpu_comparison.sh [cuda|hip]     (default cuda)
#   size comes from benchmark_config.mk
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

backend="${1:-cuda}"
[ "$backend" = cuda ] || [ "$backend" = hip ] || { echo "usage: $0 [cuda|hip]"; exit 2; }
FC="${FC:-nvfortran}"
gcc_="$(cd "$REPO/redi" && make -s print-NVCC BACKEND=$backend 2>/dev/null)"
dcver="$(fc_version "$FC")"; gcver="$(fc_version "$gcc_")"
dcflags="$(cd "$REPO/redi" && make -s print-BASE_FFLAGS FC=$FC) $(cd "$REPO/redi" && make -s print-DC_MODE_FLAGS DATA=acc FC=$FC)"
gcflags="$(cd "$REPO/redi" && make -s print-CUFLAGS BACKEND=$backend 2>/dev/null)"
dev="$(device_for dc_gpu_acc)"

banner "DC vs $backend  (per kernel; separate binaries -- see 'make cmp' for same-image)"
echo "  DC       : $dcver"
echo "  $backend : $gcver"
echo "  device   : $dev    git: $GHASH    host: $HOSTN"; echo
ensure_lock; gpu_assert_free || exit 1

csv="$DATADIR/gpu_comparison_${backend}_${TS}.csv"; csv_new "$csv"
printf '  %-22s %11s %11s %10s\n' kernel "DC ms" "$backend ms" "DC/$backend"
printf -- '  --------------------------------------------------------------\n'
for k in "${ALL_KERNELS[@]}"; do
   rm -rf "$REPO/$k/build/dc_acc" "$REPO/$k/build/cpp_$backend"
   dc=""; gc=""
   if ( cd "$REPO/$k" && make dc DATA=acc FC=$FC ) >/dev/null 2>&1; then
      dc=$(parse_ms "$( cd "$REPO/$k" && "$LOCK" "$k-cmp-dc" ./dc_acc $ARGS 2>&1 )")
   fi
   if ( cd "$REPO/$k" && make cpp BACKEND=$backend ) >/dev/null 2>&1; then
      gc=$(parse_ms "$( cd "$REPO/$k" && "$LOCK" "$k-cmp-gc" ./cpp_$backend $ARGS 2>&1 )")
   fi
   ratio=$(awk -v a="${dc:-0}" -v b="${gc:-0}" 'BEGIN{ if(b>0) printf "%.3f", a/b; else print "NaN" }')
   printf '  %-22s %11s %11s %10s\n' "$k" "${dc:-FAIL}" "${gc:-FAIL}" "$ratio"
   emit_row "$csv" "$k" dc_gpu_acc      "$FC"    "$dcver" "$dcflags" 1 "$dev" "${dc:-NaN}"
   emit_row "$csv" "$k" "gpuc_$backend" "$gcc_"  "$gcver" "$gcflags" 1 "$dev" "${gc:-NaN}"
done
echo; echo "  -> $csv"
echo "  (DC/$backend > 1 means $backend is faster; < 1 means do concurrent is faster)"
