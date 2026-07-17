#!/usr/bin/env bash
# Build + run the whole-model RK2 driver (ideal_benchmark/): all 8 ocean
# kernels back-to-back per step on a hot GPU (or CPU), timing the selected
# side(s) and version. This is the model-level number for the paper.
#
#   ./build_and_run_full_driver.sh [options]
#     --target gpu|cpu     where to run          (default gpu)
#     --mode   dc|cuda|both  which side to TIME  (default both; forced dc on cpu)
#     --version opt|unopt  optimized vs faithful (default opt)
#     --serial             cpu only: serial DC (clear -stdpar=multicore)
#     --fc <compiler>      cpu only: FC (default nvfortran)
#
#   size comes from benchmark_config.mk. Emits the RESULT lines to a CSV and
#   echoes the driver's own per-kernel table.
#
# Examples (the paper's model-level matrix):
#   ./build_and_run_full_driver.sh --target gpu --mode both --version unopt   # naive-DC vs faithful-CUDA
#   ./build_and_run_full_driver.sh --target gpu --mode both --version opt     # opt-DC   vs opt-CUDA
#   ./build_and_run_full_driver.sh --target cpu --version opt --serial        # whole model, CPU serial DC
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

target=gpu mode=both version=opt serial=0 FC=nvfortran
while [ $# -gt 0 ]; do
   case "$1" in
      --target)  target="$2"; shift 2;;
      --mode)    mode="$2";   shift 2;;
      --version) version="$2"; shift 2;;
      --serial)  serial=1; shift;;
      --fc)      FC="$2"; shift 2;;
      -h|--help) sed -n '2,20p' "$0"; exit 0;;
      *) echo "unknown option: $1" >&2; exit 2;;
   esac
done
[ "$target" = gpu ] || [ "$target" = cpu ] || { echo "target must be gpu|cpu" >&2; exit 2; }
[ "$version" = opt ] || [ "$version" = unopt ] || { echo "version must be opt|unopt" >&2; exit 2; }

bench_dir="$REPO/ideal_benchmark"

if [ "$target" = cpu ]; then
   mode=dc
   host_flags=""; [ "$serial" = 1 ] && host_flags="DC_HOST_FLAGS="
   thread="multicore"; [ "$serial" = 1 ] && thread="serial"
   banner "FULL DRIVER -- whole ocean model, CPU $thread do concurrent  (FC=$FC, version=$version)"
   ( cd "$bench_dir" && make rk2_cpu FC=$FC $host_flags ) >/dev/null 2>&1 \
      || { echo "  build rk2_cpu FAILED" >&2; exit 1; }
   out=$( cd "$bench_dir" && ./rk2_cpu $ARGS "$version" 2>&1 )
else
   [ "$mode" = dc ] || [ "$mode" = cuda ] || [ "$mode" = both ] || { echo "mode must be dc|cuda|both" >&2; exit 2; }
   banner "FULL DRIVER -- whole ocean model, GPU  (mode=$mode, version=$version)"
   ensure_lock; gpu_assert_free || exit 1
   ( cd "$bench_dir" && make rk2 ) >/dev/null 2>&1 || { echo "  build rk2 FAILED" >&2; exit 1; }
   out=$( cd "$bench_dir" && "$LOCK" "rk2-$mode-$version" ./rk2 $ARGS "$mode" "$version" 2>&1 )
fi

echo "$out"
csv="$DATADIR/full_driver_$(stamp).csv"
csv_init "$csv" "target,mode,version,nx,ny,nz,ms_per_stage,ms_per_step"
echo "$out" | grep '^RESULT' | while read -r line; do
   # RESULT target=gpu mode=dc version=opt ms_per_stage=.. ms_per_step=..
   eval "$(echo "$line" | sed 's/^RESULT //; s/ /;/g' | tr ';' '\n' | sed 's/^/R_/')" 2>/dev/null
   csv_row "$csv" "${R_target},${R_mode},${R_version},$NXP,$NYP,$NZ,${R_ms_per_stage},${R_ms_per_step}"
done
echo "  -> $csv"
