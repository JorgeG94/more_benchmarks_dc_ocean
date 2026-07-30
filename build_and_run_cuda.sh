#!/usr/bin/env bash
# Build + run every kernel's hand-written GPU-C driver. Default nvcc/CUDA; pass
# `hip` for the AMD/ROCm path (hipcc; structural on this repo's dev node). The
# paper's "gpuc_<backend>" column (the hand-CUDA rewrite that part 2 weighs
# against do concurrent). Emits the unified provenance CSV.
#
#   ./build_and_run_cuda.sh [cuda|hip]      (default cuda)
#   size comes from benchmark_config.mk
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

backend="${1:-cuda}"
[ "$backend" = cuda ] || [ "$backend" = hip ] || { echo "usage: $0 [cuda|hip]"; exit 2; }
mode="gpuc_$backend"
cc="$(cd "$REPO/redi" && make -s print-NVCC BACKEND=$backend 2>/dev/null)"
command -v "$cc" >/dev/null 2>&1 || { echo "ERROR: GPU-C compiler '$cc' not found on PATH (BACKEND=$backend)." >&2; exit 1; }
ver="$(fc_version "$cc")"
flags="$(cd "$REPO/redi" && make -s print-CUFLAGS BACKEND=$backend 2>/dev/null)"
dev="$(device_for "$mode")"

banner "hand-written GPU-C  (BACKEND=$backend)"
echo "  compiler : $ver"
echo "  flags    : $flags"
echo "  device   : $dev"
echo "  git      : $GHASH    host: $HOSTN"; echo
ensure_lock; gpu_assert_free || exit 1

csv="$DATADIR/gpuc_${backend}_${TS}.csv"; csv_new "$csv"
printf '  %-22s %-12s %14s\n' kernel build ms/step
printf -- '  ---------------------------------------------------\n'
for k in "${ALL_KERNELS[@]}"; do
   rm -rf "$REPO/$k/build/cpp_$backend"
   if out=$( cd "$REPO/$k" && make cpp BACKEND=$backend 2>&1 ); then
      run=$( cd "$REPO/$k" && "$LOCK" "$k-$mode" ./cpp_$backend $ARGS 2>&1 )
      ms=$(parse_ms "$run")
      printf '  %-22s %-12s %14s\n' "$k" "built" "${ms:-**FAIL**}"
      emit_row "$csv" "$k" "$mode" "$cc" "$ver" "$flags" 1 "$dev" "${ms:-NaN}"
   else
      ( cd "$REPO/$k" && echo "$out" > .cuda_build.log )
      printf '  %-22s %-12s %s\n' "$k" "FAILED" "see $k/.cuda_build.log"
      emit_row "$csv" "$k" "$mode" "$cc" "$ver" "$flags" 1 "$dev" NaN
   fi
done
echo; echo "  -> $csv"
