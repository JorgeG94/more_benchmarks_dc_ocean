#!/usr/bin/env bash
# probe_amdflang_dc.sh -- compile + run all 5 variants of
# amdflang_nested_record_dc.F90 and print a PASS/FAIL table.
#
# Answers two questions in one shot:
#   1. WHICH nesting shape breaks `-fdo-concurrent-to-openmp=device`
#      (V=1 scalar DT component / V=2 allocatable array of DT / V=5 control)
#   2. WHETHER either workaround dodges it (V=3 associate / V=4 array dummies)
#
# If V=3 or V=4 PASSes, that idiom is the fix to apply to the affected kernels
# (continuity_layered, kappa_shear, epbl, ale_remap) -- see NOTES_ON_PERF.md.
#
#   ./probe_amdflang_dc.sh                      # amdflang, device pass, gfx90a
#   ./probe_amdflang_dc.sh host                 # amdflang, host pass
#   ./probe_amdflang_dc.sh device gfx942        # different arch
#   FC=gfortran ./probe_amdflang_dc.sh serial   # sanity-check the source itself
#
# Expected sum for every variant: 1049600.0
set -uo pipefail
cd "$(dirname "$0")"

pass="${1:-device}"; arch="${2:-gfx90a}"
FC="${FC:-amdflang}"
SRC=amdflang_nested_record_dc.F90
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

case "$pass" in
   device) FLAGS="-fopenmp --offload-arch=$arch -fdo-concurrent-to-openmp=device" ;;
   host)   FLAGS="-fopenmp -fdo-concurrent-to-openmp=host" ;;
   serial) FLAGS="-cpp" ;;
   *)      echo "usage: $0 [device|host|serial] [arch]" >&2; exit 2 ;;
esac

command -v "$FC" >/dev/null 2>&1 || { echo "ERROR: '$FC' not on PATH." >&2; exit 1; }

# where to write .mod files -- same per-compiler spelling as ../../config.mk
case "${FC##*/}" in
   gfortran) MODFLAG=-J ;;
   nvfortran|ifx) MODFLAG=-module ;;
   *) MODFLAG=-module-dir ;;
esac

echo "=================================================================="
echo "  amdflang do-concurrent nested-record probe"
echo "  compiler : $($FC --version 2>&1 | head -1)"
echo "  pass     : $pass      flags: $FLAGS"
echo "=================================================================="
printf '  %-4s %-14s %-9s %s\n' V shape result detail
printf -- '  ---------------------------------------------------------------\n'

names=(NESTED_SCALAR NESTED_ARRAY ASSOCIATE DUMMY_ARG FLAT_CONTROL)
rc_any=0
for v in 1 2 3 4 5; do
   name="${names[$((v - 1))]}"
   log="$TMP/v$v.log"
   if "$FC" -O3 $FLAGS -DV=$v $MODFLAG "$TMP" -o "$TMP/v$v" "$SRC" > "$log" 2>&1; then
      if out=$("$TMP/v$v" 2>&1); then
         got=$(printf '%s' "$out" | grep -oE 'sum =[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+$')
         if printf '%s' "$got" | grep -qE '^1049600\.?0*$'; then
            printf '  %-4s %-14s %-9s %s\n' "$v" "$name" "PASS" "sum=$got"
         else
            printf '  %-4s %-14s %-9s %s\n' "$v" "$name" "WRONG" "sum=$got (want 1049600.0)"
            rc_any=1
         fi
      else
         printf '  %-4s %-14s %-9s %s\n' "$v" "$name" "RUNFAIL" "$(printf '%s' "$out" | head -1)"
         rc_any=1
      fi
   else
      # summarise the compiler failure: NYI message, crash, or plain error
      why=$(grep -m1 -oE 'not yet implemented:.*'                "$log")
      [ -n "$why" ] || why=$(grep -m1 -oE "instance of '[^']+'"   "$log" | sed 's/instance of/CRASH:/')
      [ -n "$why" ] || why=$(grep -m1 -iE 'error:'                "$log" | head -c 90)
      [ -n "$why" ] || why="build failed (see below)"
      printf '  %-4s %-14s %-9s %s\n' "$v" "$name" "BUILDFAIL" "$why"
      cp "$log" "./probe_v$v.log"
      rc_any=1
   fi
done

echo
echo "  Failing variants' full logs saved as ./probe_v<N>.log"
echo "  V=1/V=2 BUILDFAIL + V=3 or V=4 PASS  =>  that idiom is the kernel fix."
echo "  V=5 must PASS; if it does not, the problem is not nesting-related."
exit $rc_any
