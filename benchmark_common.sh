#!/usr/bin/env bash
# benchmark_common.sh -- shared helpers for the ./build_and_run_*.sh paper
# harness. Not executable on its own; each script `source`s it.
#
#   - reads the 5 problem-size numbers from benchmark_config.mk (the ONLY knob)
#   - the kernel list (ocean core + the coastal hll)
#   - GPU serialization lock + an exclusive-GPU guard (the ~8% effect is the
#     size of contention noise -- see NOTES_ON_PERF.md)
#   - a robust ms parser and tiny CSV writers
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$REPO/benchmark_config.mk"
LOCKDIR="$REPO/tmp_local_artifacts"
LOCK="$LOCKDIR/gpu_run.sh"
DATADIR="$REPO/paper_data"
mkdir -p "$DATADIR"

# ---- the 8 ocean-core kernels (RK2 order) + the coastal SWE flux (hll) ------
OCEAN_KERNELS=(continuity_layered redi kappa_shear ale_remap btstep epbl meke hvisc)
ALL_KERNELS=("${OCEAN_KERNELS[@]}" hll_fluxes)

# ---- problem size: five numbers, from benchmark_config.mk ONLY --------------
cfg() { awk -v k="$1" '$1==k && ($2=="?="||$2=="="){print $3; exit}' "$CFG"; }
NXP=$(cfg NXP); NYP=$(cfg NYP); NZ=$(cfg NZ); REPS=$(cfg REPS); WARM=$(cfg WARM)
: "${NXP:?benchmark_config.mk: NXP unset}" "${NYP:?}" "${NZ:?}" "${REPS:?}" "${WARM:?}"
ARGS="$NXP $NYP $NZ $REPS $WARM"

banner() {
   echo "=================================================================="
   echo "  $1"
   echo "  size: ${NXP}x${NYP}x${NZ}  reps=${REPS} warm=${WARM}   (edit benchmark_config.mk)"
   echo "=================================================================="
}

# ---- GPU lock: recreate the flock+idle-gate wrapper if missing --------------
ensure_lock() {
   [ -x "$LOCK" ] && return 0
   mkdir -p "$LOCKDIR"
   cat > "$LOCK" <<'SH'
#!/usr/bin/env bash
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
label="${1:-?}"; shift
exec 9>"$DIR/gpu.lock"; flock 9
for _ in $(seq 1 150); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ "${u:-100}" -lt 12 ] 2>/dev/null && break; sleep 2
done
printf '%s  ACQUIRE %-16s util=%s%%  cmd: %s\n' "$(date +%T)" "$label" "${u:-?}" "$*" >> "$DIR/gpu.log"
"$@"; rc=$?
printf '%s  RELEASE %-16s exit=%s\n' "$(date +%T)" "$label" "$rc" >> "$DIR/gpu.log"
exit $rc
SH
   chmod +x "$LOCK"
}

# ---- exclusive-GPU guard: the whole-model effect == contention noise --------
gpu_assert_free() {
   local n
   n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
   if [ "${n:-0}" -gt 0 ]; then
      echo "  WARNING: ${n} compute app(s) already on the GPU -- timings may be contaminated:" >&2
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/     /' >&2
      if [ "${BENCH_FORCE:-0}" != 1 ]; then
         echo "  Refusing to run. Free the GPU, or set BENCH_FORCE=1 to override." >&2
         return 1
      fi
      echo "  BENCH_FORCE=1 set -- running anyway." >&2
   fi
}

# ---- parse "NN.NNNN ms/(rep|step|call|stage)" from a kernel's stdout --------
parse_ms() {  # arg: text blob
   printf '%s\n' "$1" | grep -oE '[0-9]+\.[0-9]+[[:space:]]*ms/(rep|step|call|stage)' \
      | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1
}

# ---- build a kernel in some mode, run its plain driver, echo ms ------------
# usage: build_run <kernel_dir> "<make args>" <binary> <label> <lock:0|1>
build_run() {
   local kdir="$1" mk="$2" bin="$3" label="$4" lock="$5" out
   if ! ( cd "$REPO/$kdir" && make $mk ) >/dev/null 2>&1; then
      echo "BUILD_FAIL" >&2; echo ""; return 1
   fi
   if [ "$lock" = 1 ]; then
      out=$( cd "$REPO/$kdir" && "$LOCK" "$label" ./"$bin" $ARGS 2>&1 )
   else
      out=$( cd "$REPO/$kdir" && ./"$bin" $ARGS 2>&1 )
   fi
   parse_ms "$out"
}

csv_init() { printf '%s\n' "$2" > "$1"; }         # file, header
csv_row()  { printf '%s\n' "$2" >> "$1"; }        # file, row
stamp()    { date +%Y%m%d_%H%M%S; }
