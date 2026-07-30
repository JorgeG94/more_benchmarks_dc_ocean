#!/usr/bin/env bash
# gen_serialdo.sh -- (re)generate the COMMITTED serial-do source tree.
#
# The serial-do baseline sources are MAINTAINED IN THE REPO (committed), not
# regenerated at build time. This script refreshes them from the `do concurrent`
# sources via tools/dc_to_serialdo.py. Run it whenever a DC source changes, then
# commit the result. `--check` regenerates to a temp and fails if the committed
# tree is stale (for CI / a pre-commit guard).
#
# For each kernel, every *.F90 (and drivers/*.F90) is transformed; only files
# that actually differ (i.e. contain `do concurrent`) are written to
# <kernel>/serialdo/<basename>.F90. Byte-identical passthroughs are skipped, so
# the serialdo/ dir holds ONLY the files that genuinely change.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
XFORM="$REPO/tools/dc_to_serialdo.py"
KERNELS=(continuity_layered redi kappa_shear ale_remap btstep epbl meke hvisc hll_fluxes)
check=0; [ "${1:-}" = "--check" ] && check=1

rc=0
for k in "${KERNELS[@]}"; do
   [ -d "$REPO/$k" ] || continue
   dst="$REPO/$k/serialdo"
   for f in "$REPO/$k"/*.F90 "$REPO/$k"/drivers/*.F90; do
      [ -f "$f" ] || continue
      base="$(basename "$f")"
      tmp="$(mktemp)"
      if ! python3 "$XFORM" "$f" "$tmp" 2>/tmp/sd.err; then
         echo "  !! TRANSFORM FAIL  $k/$base : $(cat /tmp/sd.err)" >&2; rc=1; rm -f "$tmp"; continue
      fi
      if diff -q "$f" "$tmp" >/dev/null 2>&1; then
         rm -f "$tmp"; continue          # no do concurrent -> nothing to commit
      fi
      out="$dst/$base"
      if [ "$check" = 1 ]; then
         if ! diff -q "$out" "$tmp" >/dev/null 2>&1; then
            echo "  STALE  $k/serialdo/$base  (run tools/gen_serialdo.sh and commit)" >&2; rc=1
         fi
         rm -f "$tmp"
      else
         mkdir -p "$dst"
         if diff -q "$out" "$tmp" >/dev/null 2>&1; then
            echo "  up-to-date  $k/serialdo/$base"
         else
            mv "$tmp" "$out"; echo "  wrote       $k/serialdo/$base"
         fi
      fi
   done
done
[ "$check" = 1 ] && [ "$rc" = 0 ] && echo "serial-do tree is fresh."
exit $rc
