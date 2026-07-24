#!/usr/bin/env bash
# Build + run every kernel as `do concurrent` compiled SERIAL on the CPU -- no
# OpenMP, no OpenACC (DATA=none with the host parallel flag cleared). This is
# the paper's "dc_serial" baseline: the SAME do-concurrent source that runs on
# the GPU, executed as one thread. Needs a compiler that can compile
# `do concurrent ... local(...)` (nvfortran / ifx / gfortran >= 15) -- gfortran
# <= 14 cannot; use ./build_and_run_serial_do.sh for the plain-do baseline there.
#
# Contrast with ./build_and_run_serial_do.sh (mode "serial_do", plain nested
# `do`): both are single-thread CPU, but this one keeps the `do concurrent`
# construct, so the pair measures what the construct itself costs a compiler.
#
#   ./build_and_run_cpu_serial.sh [FC] [FCFLAGS]
#     FC       compiler (default nvfortran). e.g. nvfortran | ifx | gfortran
#     FCFLAGS  override the base compile flags (default: config.mk's FFLAGS_BASE)
#
#   Problem size comes from benchmark_config.mk (edit there -- no size args).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

FC="${1:-${FC:-nvfortran}}"
FCFLAGS="${2:-${FCFLAGS:-}}"
mkargs=(FC="$FC")
[ -n "$FCFLAGS" ] && mkargs+=(FFLAGS_BASE="$FCFLAGS")

# ---- provenance -----------------------------------------------------------
if ! command -v "$FC" >/dev/null 2>&1; then
   echo "ERROR: compiler '$FC' not found on PATH." >&2; exit 1
fi
ver="$(fc_version "$FC")"
flags="$(cd "$REPO/redi" && make -s print-BASE_FFLAGS "${mkargs[@]}" 2>/dev/null)"
dev="$(device_for dc_serial)"

# ---- guard: this mode needs a do-concurrent-capable compiler --------------
probe=$(mktemp -d)
cat > "$probe/p.f90" <<'F90'
program p
  integer :: i, s(4)
  do concurrent(i=1:4) local(s)
    s(i) = i
  end do
  print *, sum(s)
end program p
F90
if ! ( "$FC" -o "$probe/p" "$probe/p.f90" ) >/dev/null 2>&1; then
   echo "ERROR: $FC cannot compile 'do concurrent ... local()' -- the dc_serial mode" >&2
   echo "       is not available for this compiler. For a serial baseline here use:" >&2
   echo "         ./build_and_run_serial_do.sh $FC" >&2
   rm -rf "$probe"; exit 1
fi
rm -rf "$probe"

banner "CPU SERIAL -- do concurrent compiled serial (dc_serial)  (FC=$FC)"
echo "  compiler : $ver"
echo "  flags    : $flags"
echo "  device   : $dev"
echo "  git      : $GHASH    host: $HOSTN"
echo

csv="$DATADIR/cpu_serial_${FC##*/}_${TS}.csv"; csv_new "$csv"
printf '  %-22s %-12s %14s\n' kernel build ms/step
printf -- '  ---------------------------------------------------\n'
for k in "${ALL_KERNELS[@]}"; do
   rm -rf "$REPO/$k/build/dc_none"     # fresh build: never reuse another compiler/mode's objects
   if out=$( cd "$REPO/$k" && make dc DATA=none DC_HOST_FLAGS= "${mkargs[@]}" 2>&1 ); then
      run=$( cd "$REPO/$k" && OMP_NUM_THREADS=1 ./dc_none $ARGS 2>&1 )
      ms=$(parse_ms "$run")
      printf '  %-22s %-12s %14s\n' "$k" "built" "${ms:-**FAIL**}"
      emit_row "$csv" "$k" dc_serial "$FC" "$ver" "$flags" 1 "$dev" "${ms:-NaN}"
   else
      ( cd "$REPO/$k" && echo "$out" > .cpu_serial_build.log )
      printf '  %-22s %-12s %s\n' "$k" "FAILED" "see $k/.cpu_serial_build.log"
      emit_row "$csv" "$k" dc_serial "$FC" "$ver" "$flags" 1 "$dev" NaN
   fi
done
echo; echo "  -> $csv"
