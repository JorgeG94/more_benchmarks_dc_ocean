#!/usr/bin/env bash
# tools/ks_min.sh -- the kappa_shear depth sweep with ZERO probing.
#
# tools/ks_sweep.sh discovers everything: which compilers exist, which devices,
# what architecture, whether the GPU is idle. That is the right behaviour on an
# unfamiliar machine and the wrong behaviour on a known one -- `nvidia-smi` costs
# seconds per call on Grace-Hopper, and the full harness calls it once per
# emitted row plus a polling idle-gate before EVERY run.
#
# This script runs nvidia-smi ZERO times. Every fact it needs is stated in
# tools/ks_min.conf (or the environment). It takes no lock and detects nothing.
#
#   ./tools/ks_min.sh                                   # uses ks_min.conf
#   MODE=dc_gpu DATA=omp ./tools/ks_min.sh              # OpenMP-target lane
#   FC=ifx MODE=dc_gpu DATA=omp \
#     DC_GPU_FLAGS='-qopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent -DDC_DATA_OMP' \
#     DEVICE='Intel GPU' ./tools/ks_min.sh
#   MODE=dc_multicore FC=flang ./tools/ks_min.sh        # thread sweep
#   CUDA=on ./tools/ks_min.sh                           # + the CUDA head-to-head
#   ./tools/ks_min.sh -n                                # dry run: print the plan
#
# Output is the SAME CSV schema tools/ks_sweep.sh writes, so tools/ks_report.py
# reads both and rows from different machines aggregate.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$here/.." && pwd)"
KDIR="$REPO/kappa_shear"
# GNU make, under whatever name it has. Apple ships GNU make 3.81 (2006) as
# `make`; `brew install make` provides a current one as `gmake`. Overridable:
#   MAKE=gmake ./tools/ks_min.sh
MAKE="${MAKE:-make}"
CONF="${KS_MIN_CONF:-$here/ks_min.conf}"
[ -f "$CONF" ] || { echo "missing config: $CONF" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONF"

DRY=0
[ "${1:-}" = "-n" ] || [ "${1:-}" = "--dry-run" ] && DRY=1

# ---------------------------------------------------------------------------
# Normalise the on/off knobs, and REJECT anything unrecognised.
#
# `[ "$CUDA" = on ]` is case-sensitive, so `CUDA=ON` silently selected the off
# path -- while the banner printed "CUDA=ON" and the run looked like it had done
# what was asked. A knob that quietly does nothing is the worst failure mode in
# a benchmark harness: it produces a complete, plausible CSV with a whole lane
# missing. Accept the obvious spellings; refuse the rest loudly.
# ---------------------------------------------------------------------------
norm_bool() {
   case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
      on|1|yes|true|y)      echo on;;
      off|0|no|false|n|"")  echo off;;
      *)                    echo invalid;;
   esac
}
for _k in CUDA COUNTERS; do
   _raw="$(eval printf '%s' "\$$_k")"
   _v="$(norm_bool "$_raw")"
   if [ "$_v" = invalid ]; then
      echo "ERROR: $_k='$_raw' is not a boolean (use on|off, 1|0, yes|no, true|false)" >&2
      exit 2
   fi
   eval "$_k=$_v"
done

# The per-thread column frame is stack-resident; at a deep NZSTACK it exceeds a
# default 8 MB thread stack and the failure looks like a code bug.
ulimit -s unlimited 2>/dev/null || true

OUT="$REPO/paper_data"
mkdir -p "$OUT"
TS="$(date +%Y%m%d_%H%M%S)"
HOSTN="$(hostname)"
GHASH="$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo nogit)"
CSV="$OUT/ks_min_${HOSTN}_${TS}.csv"

KS_COLS="experiment,mode,impl,launcher,variant,lanes,compiler,compiler_ver,flags,\
stack_policy,copy_policy,phys,nzstack,bcopy,opt_nzmax,minblocks,\
threads,proc_bind,device,gpu_arch,nxp,nyp,nx,ny,ncols,nwet,nz,land_pct,\
reps,warm,nrun,ms_min,ms_med,ns_per_col,it_outer,it_inner,\
kd_sum,kd_min,kd_max,regs,local_b,status,git_hash,host,timestamp"

FCVER="$("$FC" --version 2>&1 | grep -m1 . | tr ',' ' ')"
_q() { printf '%s' "$1" | tr '\n"' " '"; }

# ---------------------------------------------------------------------------
# make args for the selected MODE. The kernel Makefile owns every binary name,
# so ask it rather than reconstructing them here.
# ---------------------------------------------------------------------------
MKVARS=(); MKTARGET=""; BINVAR=""
# FLAGVARS: which make variable(s) actually spell the compile line for THIS
# mode. Not cosmetic -- the CSV's `flags` column is the paper's provenance, and
# hardcoding BASE_FFLAGS+DC_MODE_FLAGS made a gfortran serial-do build on a Mac
# record `-stdpar=gpu -acc=gpu -gpu=cc90,mem:separate`: the serialdo path
# compiles with SD_FFLAGS and never sees a data layer, and with no DATA= in
# MKVARS the Makefile's DC_MODE_FLAGS defaults to the GPU one. Timings were
# never affected; the record of how they were produced was.
set_make_args() {   # $1 = nzstack, $2 = counters(0|1)
   local nzs="$1" cnt="$2"
   MKVARS=(FC="$FC" ARCH="$ARCH" NVARCH="$NVARCH" NZSTACK="$nzs" CNT="$cnt"
           BCOPY="$BCOPY")
   case "$MODE" in
      serial_do)    MKTARGET=serialdo; BINVAR=SDBIN; FLAGVARS="SD_FFLAGS";;
      dc_serial)    MKTARGET=dc; BINVAR=DCBIN; FLAGVARS="BASE_FFLAGS DC_MODE_FLAGS"
                    MKVARS+=(DATA=none DC_HOST_FLAGS=);;
      dc_multicore) MKTARGET=dc; BINVAR=DCBIN; FLAGVARS="BASE_FFLAGS DC_MODE_FLAGS"
                    MKVARS+=(DATA=none);;   # config.mk supplies the host flag
      dc_gpu)       MKTARGET=dc; BINVAR=DCBIN; FLAGVARS="BASE_FFLAGS DC_MODE_FLAGS"
                    MKVARS+=(DATA="$DATA")
                    [ -n "$DC_GPU_FLAGS" ] && MKVARS+=("DC_MODE_FLAGS=$DC_GPU_FLAGS");;
      *) echo "MODE must be serial_do | dc_serial | dc_multicore | dc_gpu" >&2; exit 2;;
   esac
}
# The CUDA side's FULLCOPY must be the MIRROR of the Fortran side's BCOPY --
# see kappa_shear/Makefile, "THIS IS THE 1:1 PAIRING KNOB":
#   BCOPY=0 (Fortran copies O(NZSTACK))  <->  FULLCOPY=1 (CUDA copies O(NZSTACK))
#   BCOPY=1 (Fortran copies O(nz))       <->  FULLCOPY=0 (CUDA copies O(nz))
# Both default to 0 in the Makefile, which is the MISMATCHED diagonal: Fortran
# copying 128 doubles per substep against CUDA copying nz. That does not measure
# codegen, it measures a copy the two sides were not asked to do equally -- and
# it inflates the dc/cuda ratio hardest at shallow nz, where NZSTACK/nz is
# largest. Derived here rather than passed, so the two can never drift apart.
set_cmp_args() {    # $1 = nzstack, $2 = counters
   MKVARS=(FC="$FC" ARCH="$ARCH" NVARCH="$NVARCH" NZSTACK="$1" CNT="$2"
           BCOPY="$BCOPY" FULLCOPY="$((1 - BCOPY))" OPT_NZMAX="$1")
   MKTARGET=cmp; BINVAR=CMPBIN; FLAGVARS="CMP_FFLAGS CMP_CUFLAGS"
}
FLAGVARS="BASE_FFLAGS DC_MODE_FLAGS"
# What the three whole-array assignments in ks_solve_column actually copied, and
# -- for a cmp run -- whether the CUDA side was asked to copy the same amount.
# This column used to be the literal string "prod" in every row ever written, so
# the one axis that decides whether a dc-vs-cuda ratio means anything was not
# recorded at all.
copy_policy() {
   local f=full; [ "$BCOPY" = 1 ] && f=bounded
   [ "${USING_CMP:-0}" = 1 ] && printf 'paired-%s' "$f" || printf '%s' "$f"
}
# the resolved compile line for the build that was actually run
flags_str() { local v out=""; for v in $FLAGVARS; do out="$out $(mkprint "$v")"; done
              printf '%s' "${out# }"; }
mkprint() { ( cd "$KDIR" && "$MAKE" -s "print-$1" "${MKVARS[@]}" 2>/dev/null ); }

# ---------------------------------------------------------------------------
# run one binary; echo stdout. No lock, no idle gate, no device query.
# ---------------------------------------------------------------------------
run_bin() {   # $1 bin, $2 nz, $3 reps, $4 warm, $5 threads, $6 nzstack
   local frame_mb=$(( (60 * ($6 + 1) * 8 + 1048575) / 1048576 ))
   local stk=$(( frame_mb * 8 )); [ "$stk" -lt 64 ] && stk=64
   ( cd "$KDIR" && env KS_TARGET_MS="$KS_TARGET_MS" KS_MAX_REPS="$KS_MAX_REPS" \
        KS_PHYS="$KS_PHYS" \
        OMP_NUM_THREADS="$5" ACC_NUM_CORES="$5" \
        OMP_PROC_BIND=close OMP_PLACES=cores OMP_STACKSIZE="${stk}M" \
        ./"$1" "$NXP" "$NYP" "$2" "$3" "$4" "$LAND_PCT" 2>&1 )
}

# Registers + per-thread frame for the `do concurrent` kernel, read out of the
# built binary. THE cross-architecture diagnostic: the CUDA side reports this at
# run time via cudaFuncGetAttributes, but the Fortran side has no equivalent, so
# without this the most informative column is blank for exactly the half of the
# comparison under investigation. Measured on V100/nvfortran 26.5: DC sits at the
# 254-register cap in every build while nvcc uses 80. Local, cheap, no device query.
CUOBJDUMP="$(command -v cuobjdump 2>/dev/null)"
[ -n "$CUOBJDUMP" ] || CUOBJDUMP="$(dirname "$(command -v nvfortran 2>/dev/null)" 2>/dev/null)/cuobjdump"
[ -x "$CUOBJDUMP" ] || CUOBJDUMP=""
gpu_res_usage() {   # $1 = binary (relative to KDIR); prints "REG STACK"
   [ -n "$CUOBJDUMP" ] && [ -f "$KDIR/$1" ] || { echo ""; return; }
   "$CUOBJDUMP" -res-usage "$KDIR/$1" 2>/dev/null | awk '
      /^ Function /{ f=$2; sub(/:$/,"",f) }
      /REG:/{ r=0; st=0
         if (match($0,/REG:[0-9]+/))   r  = substr($0,RSTART+4,RLENGTH-4)+0
         if (match($0,/STACK:[0-9]+/)) st = substr($0,RSTART+6,RLENGTH-6)+0
         # nvfortran names device kernels <module>_<procedure>_<line>_gpu. Match
         # the SUFFIX and take the biggest frame rather than a specific
         # procedure name: pinning the name to `column_kernel` silently returned
         # nothing the moment the loop was refactored into ks_column_loop, and a
         # blank column looks exactly like "the tool is not installed".
         # The nvcc kernels are C++-mangled (_Z...) so they never match _gpu.
         if (f ~ /_gpu$/ && st > bs) { br=r; bs=st } }
      END{ if (br+0>0) print br, bs }'
}

# FIELD ACCESS WITHOUT ASSOCIATIVE ARRAYS. stock macOS ships bash 3.2, which has
# no `declare -A` -- the script would die at parse time before running anything.
# RESULT lines are `key=value` tokens, so pull one field at a time instead of
# building a map.
# One pass over the line into plain variables -- no subprocesses, no arrays.
R_KEYS="impl ms variant lanes bcopy phys nx ny ncols nwet nz reps kd_sum kd_min kd_max regs local_b it_outer it_inner"
parse_line() {
   local t k v
   for k in $R_KEYS; do eval "R_$k="; done
   for t in $1; do
      case "$t" in *=*) ;; *) continue;; esac
      k=${t%%=*}; v=${t#*=}
      case " $R_KEYS " in *" $k "*) eval "R_$k=\$v";; esac
   done
   [ -n "$R_ms" ]
}
# Field, with an optional default. The `${2:-}` is load-bearing: under `set -u`
# a bare `$2` inside the eval is an unbound variable, the eval aborts, and rv
# prints NOTHING -- so every one-argument call silently emitted an empty CSV
# column while the two-argument calls looked fine.
rv() { local _d=${2:-}; eval "printf '%s' \"\${R_$1:-\$_d}\""; }
# min/median in awk, not `sort -g`: BSD sort (macOS) is not dependable for
# general-numeric, and these values are in E notation.
amin() { awk '{if(NR==1||$1<m)m=$1} END{if(NR)printf "%.10E", m}'; }
amed() { awk '{v[NR]=$1} END{ if(!NR)exit; n=asort_done=0;
          for(i=1;i<=NR;i++)for(j=i+1;j<=NR;j++)if(v[j]<v[i]){t=v[i];v[i]=v[j];v[j]=t}
          printf "%.10E", (NR%2)? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2 }'; }

NROWS=0
emit() {  # $1 impl $2 status $3 ms_min $4 ms_med $5 nspc $6 it_out $7 it_in $8 nzs $9 spol $10 threads
   local mode_out="$1" launcher=solo
   [ "$1" = dc ] && mode_out="$MODE"
   [ "${USING_CMP:-0}" = 1 ] && { launcher=cmp; [ "$1" = dc ] && mode_out=dc_gpu; }
   printf '%s,%s,%s,%s,%s,%s,%s,"%s","%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
     "nz" "$mode_out" "$1" "$launcher" "$(rv variant faithful)" "$(rv lanes 1)" \
     "$FC" "$(_q "$FCVER")" "$(_q "$(flags_str)")" \
     "$9" "$(copy_policy)" "$(rv phys faithful)" "$8" "$(rv bcopy 0)" "$8" "0" \
     "${10}" "close" "$(_q "$DEVICE")" "$GPU_ARCH_LABEL" \
     "$NXP" "$NYP" "$(rv nx)" "$(rv ny)" "$(rv ncols)" "$(rv nwet)" \
     "$(rv nz)" "$LAND_PCT" "$(rv reps)" "$KS_WARM" "$NRUN" \
     "$3" "$4" "$5" "$6" "$7" \
     "$(rv kd_sum)" "$(rv kd_min)" "$(rv kd_max)" "$(rv regs)" "$(rv local_b)" \
     "$2" "$GHASH" "$HOSTN" "$TS" >> "$CSV"
   NROWS=$((NROWS+1))
}

# ---------------------------------------------------------------------------
# one configuration: build, NRUN timed invocations, optional counters pass
# ---------------------------------------------------------------------------
measure() {   # $1 nz, $2 nzstack, $3 stack_policy, $4 threads
   local nz="$1" nzs="$2" spol="$3" thr="$4" label bin out i line imp msv
   local tag="$MODE"; [ "${USING_CMP:-0}" = 1 ] && tag="cmp"
   label=$(printf '%-13s nz=%-4s stk=%-4s thr=%-3s' "$tag" "$nz" "$nzs" "$thr")
   if [ "$DRY" = 1 ]; then echo "    $label"; NROWS=$((NROWS+1)); return; fi

   if [ "${USING_CMP:-0}" = 1 ]; then set_cmp_args "$nzs" 0; else set_make_args "$nzs" 0; fi
   bin="$(mkprint "$BINVAR")"
   if ! ( cd "$KDIR" && "$MAKE" "$MKTARGET" "${MKVARS[@]}" ) >"$KDIR/.ks_min_build.log" 2>&1; then
      echo "    $label  BUILD_FAIL (see kappa_shear/.ks_min_build.log)"
      parse_line ""; emit dc BUILD_FAIL NaN NaN NaN NaN NaN "$nzs" "$spol" "$thr"; return
   fi

   local order=""
   for _i in dc cuda_faithful cuda_opt; do eval "MSL_$_i=; RLINE_$_i="; done
   for i in $(seq 1 "$NRUN"); do
      out=$(run_bin "$bin" "$nz" 0 "$KS_WARM" "$thr" "$nzs")
      if [ $? -ne 0 ]; then
         echo "    $label  RUN_FAIL"; printf '%s\n' "$out" > "$KDIR/.ks_min_run.log"
         parse_line ""; emit dc RUN_FAIL NaN NaN NaN NaN NaN "$nzs" "$spol" "$thr"; return
      fi
      while IFS= read -r line; do
         parse_line "$line" || continue
         imp="$(rv impl dc)"; msv="$R_ms"
         case " $order " in *" $imp "*) ;; *) order="$order $imp";; esac
         eval "MSL_$imp=\"\$MSL_$imp $msv\"; RLINE_$imp=\$line"
      done < <(printf '%s\n' "$out" | grep '^RESULT ')
   done
   [ -n "$order" ] || { echo "    $label  NO_RESULT"; parse_line ""; \
        emit dc NO_RESULT NaN NaN NaN NaN NaN "$nzs" "$spol" "$thr"; return; }

   # counters pass: integers only, never timed (this kernel is register-bound)
   local dc_ito="" dc_iti=""
   if [ "$COUNTERS" = on ]; then
      if [ "${USING_CMP:-0}" = 1 ]; then set_cmp_args "$nzs" 1; else set_make_args "$nzs" 1; fi
      local cbin; cbin="$(mkprint "$BINVAR")"
      if ( cd "$KDIR" && "$MAKE" "$MKTARGET" "${MKVARS[@]}" ) >/dev/null 2>&1; then
         while IFS= read -r line; do
            parse_line "$line" || continue
            [ "$(rv impl dc)" = dc ] && { dc_ito="$R_it_outer"; dc_iti="$R_it_inner"; }
         done < <(run_bin "$cbin" "$nz" 1 0 "$thr" "$nzs" | grep '^RESULT ')
      fi
      if [ "${USING_CMP:-0}" = 1 ]; then set_cmp_args "$nzs" 0; else set_make_args "$nzs" 0; fi
   fi

   # DC-kernel registers/frame, only meaningful for a GPU build
   local DC_REGS="" DC_STACK=""
   case "$MODE" in dc_gpu) read -r DC_REGS DC_STACK <<<"$(gpu_res_usage "$bin")";; esac
   [ "${USING_CMP:-0}" = 1 ] && read -r DC_REGS DC_STACK <<<"$(gpu_res_usage "$bin")"

   local mn md nspc ito iti
   for imp in $order; do
      eval "_ms=\$MSL_$imp; _ln=\$RLINE_$imp"
      mn=$(printf '%s\n' $_ms | amin)
      md=$(printf '%s\n' $_ms | amed)
      parse_line "$_ln" || continue
      nspc=$(awk -v m="$mn" -v n="$(rv ncols 1)" 'BEGIN{printf "%.6f", m*1e6/n}')
      ito="$(rv it_outer)"; iti="$(rv it_inner)"
      [ "$imp" = dc ] && [ -n "$dc_iti" ] && { ito="$dc_ito"; iti="$dc_iti"; }
      # the Fortran driver cannot report its own kernel's registers; cuobjdump can
      [ "$imp" = dc ] && [ -n "$DC_REGS" ] && { R_regs="$DC_REGS"; R_local_b="$DC_STACK"; }
      printf '    %s  %-14s %12s ms  %11s ns/col  it=%s\n' "$label" "$imp" "$mn" "$nspc" "${iti:-?}"
      emit "$imp" OK "$mn" "$md" "$nspc" "$ito" "$iti" "$nzs" "$spol" "$thr"
   done
}

# ---------------------------------------------------------------------------
echo "=================================================================="
echo "  kappa_shear MINIMAL sweep -- no probing, no nvidia-smi, no lock"
echo "  host    : $HOSTN   git=$GHASH"
echo "  compiler: $FC   ($FCVER)"
echo "  target  : ARCH=$ARCH NVARCH=$NVARCH device='$DEVICE'"
echo "  mode    : $MODE${MODE:+ }$( [ "$MODE" = dc_gpu ] && echo "DATA=$DATA" )  CUDA=$CUDA"
[ -n "$DC_GPU_FLAGS" ] && echo "  vendor  : DC_MODE_FLAGS='$DC_GPU_FLAGS'"
echo "  problem : ${NXP}x${NYP}, nz={$NZ_LIST}, stack={$STACK_POLICIES}, land=${LAND_PCT}%"
echo "  timing  : NRUN=$NRUN target=${KS_TARGET_MS}ms reps<=$KS_MAX_REPS warm=$KS_WARM counters=$COUNTERS"
[ "$DRY" = 1 ] && echo "  *** DRY RUN ***"
echo "=================================================================="
[ "$DRY" = 1 ] || { printf '%s\n' "$KS_COLS" > "$CSV"; echo "  -> $CSV"; echo; }

THREADS_TO_RUN="1"
[ "$MODE" = dc_multicore ] && THREADS_TO_RUN="$THREAD_LIST"

USING_CMP=0
for thr in $THREADS_TO_RUN; do
   for spol in $STACK_POLICIES; do
      for nz in $NZ_LIST; do
         case "$spol" in
            prod) [ $((nz + 1)) -le 128 ] || { echo "    skip nz=$nz stack=prod (nz+1 > 128)"; continue; }
                  nzs=128;;
            fit)  nzs=$((nz + 1));;
            # A bare NUMBER is an explicit NZ_STACK_MAX. That is what lets you
            # sweep the FRAME at fixed nz -- e.g. STACK_POLICIES="33 48 64 96 128"
            # at NZ_LIST=30 -- to find where the cost/instability sets in,
            # rather than only knowing the two endpoints.
            ''|*[!0-9]*) echo "stack policy must be prod | fit | a number (got '$spol')" >&2; exit 2;;
            *)    nzs="$spol"
                  [ "$nzs" -ge $((nz + 1)) ] || { echo "    skip nz=$nz stack=$nzs (needs >= nz+1)"; continue; };;
         esac
         measure "$nz" "$nzs" "$spol" "$thr"
      done
   done
done

if [ "$CUDA" = on ]; then
   echo; echo "  --- CUDA head-to-head (make cmp: dc + cuda_faithful + cuda_opt) ---"
   USING_CMP=1
   for spol in $STACK_POLICIES; do
      for nz in $NZ_LIST; do
         case "$spol" in
            prod) [ $((nz + 1)) -le 128 ] || continue; nzs=128;;
            fit)  nzs=$((nz + 1));;
         esac
         measure "$nz" "$nzs" "$spol" 1
      done
   done
fi

echo
echo "=================================================================="
if [ "$DRY" = 1 ]; then
   echo "  dry run: $NROWS configurations"
else
   echo "  done -> $CSV"
   echo "  rows: $NROWS   ($(grep -c ',OK,' "$CSV" 2>/dev/null || echo 0) OK)"
   echo "  next: python3 tools/ks_report.py $CSV"
fi
echo "=================================================================="
