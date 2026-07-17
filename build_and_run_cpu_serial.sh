#!/usr/bin/env bash
# Build + run every kernel as plain `do concurrent` on the CPU -- NO OpenMP,
# NO OpenACC, serial (DATA=none with the stdpar host flag cleared). The
# portability baseline: same source that runs on the GPU, on a CPU, any
# F2018 compiler. Needs no GPU and no CUDA.
#
#   ./build_and_run_cpu_serial.sh [FC]      FC = nvfortran (default) | gfortran | ifx
#   size comes from benchmark_config.mk (edit there; there are no size args)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

FC="${1:-nvfortran}"
banner "CPU SERIAL -- do concurrent, no OpenMP/OpenACC  (FC=$FC)"
csv="$DATADIR/cpu_serial_${FC##*/}_$(stamp).csv"
csv_init "$csv" "kernel,compiler,nx,ny,nz,ms_per_step"

printf '  %-22s %14s\n' kernel ms/step
for k in "${ALL_KERNELS[@]}"; do
   ms=$(build_run "$k" "dc DATA=none FC=$FC DC_HOST_FLAGS=" dc_none "$k-cpuser" 0)
   printf '  %-22s %14s\n' "$k" "${ms:-**FAIL**}"
   csv_row "$csv" "$k,$FC,$NXP,$NYP,$NZ,${ms:-NaN}"
done
echo "  -> $csv"
