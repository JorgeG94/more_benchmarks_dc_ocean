#!/usr/bin/env bash
# tools/ks_sweep.sh -- the kappa_shear size/arrangement sweep.
#
# WHY THIS EXISTS (and why run_all.sh is not enough): run_all.sh measures ONE
# problem size across modes and compilers. The kappa_shear questions are about
# how the picture MOVES with size and arrangement, and every interesting axis
# here is COMPILE-TIME -- NZ_STACK_MAX, the whole-array copy bound, the CUDA
# frame bound, the lane count. So this is a build x run matrix driver: it asks
# the Makefile for each configuration's build dir and binary name (`make
# print-DCBIN`), so build caching is automatic and the Makefile stays the single
# source of truth for what a configuration is called.
#
# PORTABLE BY CONSTRUCTION: which compilers and devices exist is detected, not
# configured. The same invocation is meant to run unmodified on a V100 box, a
# GH200, an MI250X node, an Intel GPU, or a 104-core Sapphire Rapids -- each
# reports the subset it can do and SKIPS the rest with a printed reason. Rows
# aggregate across machines by (experiment, mode, compiler, device).
#
# USAGE
#   ./tools/ks_sweep.sh                    # the default experiments (see .conf)
#   ./tools/ks_sweep.sh nz                 # just the depth sweep
#   ./tools/ks_sweep.sh --dry-run nz cols  # print the matrix + row count, run nothing
#   ./tools/ks_sweep.sh --list             # available experiments
#   NZ_LIST="30 75" COMPILERS=ifx ./tools/ks_sweep.sh nz
#   MODES=dc ./tools/ks_sweep.sh nz threads   # DC lanes only, no CUDA/HIP
#
# EXPERIMENTS
#   nz       depth sweep x stack policy -- the headline curve
#   copy     the O(NZSTACK)-vs-O(nz) whole-array copy, isolated
#   cols     parallel-width sweep -- occupancy, thread scaling, MOM6 per-rank
#   threads  dc_multicore thread scaling x OMP_PROC_BIND
#   vlen     VARIANT=block lane sweep (needs ks_block.F90)
#   cuopt    CUDA-side optimisation knobs (OPT_NZMAX / MINBLOCKS)
#   tripcount  -gpu=tripcount:host on/off -- the NVHPC TPR #38714 check.
#            RUN THIS FIRST ON ANY NEW GPU/COMPILER. Without the flag the
#            DC side is timed ~2x wrong, which looks exactly like a large
#            architecture-dependent CUDA win.
#
# OUTPUT: paper_data/ks_sweep_<host>_<ts>.csv, one row per configuration,
# carrying full provenance plus kd_sum/kd_min/kd_max as a correctness
# fingerprint -- any lane that quietly changes the answers is visible in the
# table without a separate verification run.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$here/.." && pwd)"
source "$REPO/benchmark_common.sh"     # LOCK, gpu_assert_free, _q, GHASH, HOSTN, ...
source "$REPO/tools/ks_sweep.conf"
KDIR="$REPO/kappa_shear"

# The per-thread column frame is stack-resident on the CPU; at a deep NZSTACK
# times a wide VLEN it comfortably exceeds a default 8 MB thread stack, and the
# failure mode is a segfault that looks like a code bug. Raise both limits.
ulimit -s unlimited 2>/dev/null || true

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ALL_EXPERIMENTS="nz copy cols threads vlen cuopt tripcount"
DRY=0
WANT=()
while [ $# -gt 0 ]; do
   case "$1" in
      --dry-run|-n) DRY=1;;
      --list|-l) echo "experiments: $ALL_EXPERIMENTS"; exit 0;;
      -h|--help) sed -n '2,40p' "$0" | sed 's/^# \?//'; exit 0;;
      -*) echo "unknown option $1" >&2; exit 2;;
      *) WANT+=("$1");;
   esac
   shift
done
[ ${#WANT[@]} -gt 0 ] || WANT=($EXPERIMENTS)
for e in "${WANT[@]}"; do
   case " $ALL_EXPERIMENTS " in *" $e "*) ;; *) echo "unknown experiment '$e' (have: $ALL_EXPERIMENTS)" >&2; exit 2;; esac
done

# ---------------------------------------------------------------------------
# make helpers. `print-%` lives in common/serialdo.mk, included by every kernel
# Makefile, so the Makefile itself resolves every configuration-dependent name.
# ---------------------------------------------------------------------------
MKVARS=()      # VAR=VALUE list for the current configuration
MKTARGET=""    # make target for the current mode
ARCHVARS=()    # ARCH/NVARCH, appended to every configuration (empty on a CPU box)
mkprint() { ( cd "$KDIR" && make -s "print-$1" "${MKVARS[@]}" 2>/dev/null ); }
# a print that only needs FC (used during detection, before a config exists)
mkprint_host() { ( cd "$KDIR" && make -s "print-$1" "FC=$2" 2>/dev/null ); }

mode_class() {
   case "$1" in
      serial_do)            echo serial_do;;
      dc_serial)            echo serial;;
      dc_multicore)         echo multicore;;
      dc_gpu_*)             echo gpu;;
      cmp)                  echo cmp;;
   esac
}
mode_locks()  { case "$(mode_class "$1")" in gpu|cmp) echo 1;; *) echo 0;; esac; }

# THERE IS NO C++ AXIS. The hand-written CUDA kernels are launched FROM the
# Fortran driver (`make cmp`, -DKS_WITH_CUDA + host_data use_device), so
# `do concurrent`, cuda_faithful and cuda_opt run on the SAME device allocation
# against the SAME state, in one process. That kills the C++ main()'s
# hand-mirrored build_state -- a mirror that drifts measures a different problem
# silently -- and the cross-language libm question with it. One cmp invocation
# therefore yields THREE rows, distinguished by the RESULT line's `impl=` field.
# `make cpp` still exists as the historical no-Fortran control; the sweep does
# not use it.
#
# A row's `mode` is the impl, except impl=dc which takes the lane's own mode, so
# cmp-derived DC rows stack with the standalone dc_gpu_acc rows. `launcher`
# records which binary produced the row (solo | cmp).
mode_for_impl() {
   case "$1" in
      dc) case "$cfg_mode" in cmp) echo dc_gpu_acc;; *) echo "$cfg_mode";; esac;;
      *)  echo "$1";;
   esac
}

# ---------------------------------------------------------------------------
# Detection -- compilers, devices, and therefore modes
# ---------------------------------------------------------------------------
# Can this compiler compile `do concurrent ... local(...)`? gfortran <= 14
# cannot, which is why its only lane is serial_do.
dc_capable() {
   local fc="$1" d; d=$(mktemp -d)
   cat > "$d/p.f90" <<'F90'
program p
  integer :: i, s(4)
  do concurrent(i=1:4) local(s)
    s(i) = i
  end do
  print *, sum(s)
end program p
F90
   ( "$fc" -o "$d/p" "$d/p.f90" ) >/dev/null 2>&1; local rc=$?
   rm -rf "$d"; return $rc
}

if [ "$COMPILERS" = auto ]; then
   FCS=()
   for fc in nvfortran ifx flang flang-new amdflang amdflang-new gfortran; do
      command -v "$fc" >/dev/null 2>&1 && FCS+=("$fc")
   done
else
   FCS=($COMPILERS)
fi
[ ${#FCS[@]} -gt 0 ] || { echo "no Fortran compiler on PATH" >&2; exit 1; }

have_nvidia=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && have_nvidia=1
have_amd=0;    command -v rocminfo   >/dev/null 2>&1 && have_amd=1
have_intel=0;  command -v sycl-ls    >/dev/null 2>&1 && have_intel=1
# The GPU-C compiler is whatever config.mk points NVCC at -- ask make rather
# than guessing from PATH, or the harness records a different nvcc version from
# the one that actually built the kernel. (config.mk may leave a trailing
# space; strip it.)
# GPU ARCHITECTURE. Detected once here and passed to every make invocation, so
# (a) no nvidia-smi call per make, and (b) a run on an A100/H100/GH200 cannot
# silently compile for Volta and JIT -- which would invalidate the very codegen
# ratio this sweep exists to measure. Override with GPU_ARCH_OVERRIDE=cc90.
GPU_CC="$(command -v nvidia-smi >/dev/null 2>&1 && \
          nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d ' .')"
if [ -n "${GPU_ARCH_OVERRIDE:-}" ]; then
   CFG_ARCH="$GPU_ARCH_OVERRIDE"
   CFG_NVARCH="sm_${GPU_ARCH_OVERRIDE#cc}"
   if [ -n "$GPU_CC" ] && [ "$CFG_ARCH" != "cc$GPU_CC" ]; then
      echo "  WARNING: GPU_ARCH_OVERRIDE=$CFG_ARCH but the device reports cc$GPU_CC." >&2
      echo "           Compiling for a different target than you are running on makes" >&2
      echo "           every ratio in the output suspect. Continuing as instructed." >&2
   fi
elif [ -n "$GPU_CC" ]; then
   CFG_ARCH="cc$GPU_CC"; CFG_NVARCH="sm_$GPU_CC"
else
   CFG_ARCH=""; CFG_NVARCH=""      # no GPU: leave the Makefile defaults alone
fi

# cuobjdump: the ONLY way to see what nvfortran actually allocated for the
# `do concurrent` kernel. The CUDA side reports regs/local at run time via
# cudaFuncGetAttributes, but the Fortran side has no equivalent -- so without
# this the most diagnostic column in the table is blank for exactly the half of
# the comparison under investigation. Ships with both CUDA and NVHPC.
CUOBJDUMP="$(command -v cuobjdump 2>/dev/null)"
if [ -z "$CUOBJDUMP" ]; then
   for c in "$(dirname "$(command -v nvfortran 2>/dev/null)" 2>/dev/null)/cuobjdump" \
            /usr/local/cuda/bin/cuobjdump; do
      [ -x "$c" ] && { CUOBJDUMP="$c"; break; }
   done
fi

NVCC_BIN="$( cd "$KDIR" && make -s print-NVCC 2>/dev/null | tr -d '[:space:]' )"
[ -n "$NVCC_BIN" ] || NVCC_BIN=nvcc
have_nvcc=0;   command -v "$NVCC_BIN" >/dev/null 2>&1 && have_nvcc=1
have_hipcc=0;  command -v hipcc >/dev/null 2>&1 && have_hipcc=1

if [ -n "$CFG_ARCH" ]; then ARCHVARS=(ARCH="$CFG_ARCH" NVARCH="$CFG_NVARCH"); fi

declare -A DCCAP
for fc in "${FCS[@]}"; do
   if dc_capable "$fc"; then DCCAP[$fc]=1; else DCCAP[$fc]=0; fi
done

# (mode, compiler) pairs this machine can actually run.
LANES=()   # "mode:fc"
add_lane() { LANES+=("$1:$2"); }
for fc in "${FCS[@]}"; do
   add_lane serial_do "$fc"
   [ "${DCCAP[$fc]}" = 1 ] || continue
   add_lane dc_serial "$fc"
   [ -n "$(mkprint_host DC_HOST_FLAGS "$fc")" ] && add_lane dc_multicore "$fc"
   case "${fc##*/}" in
      nvfortran) [ $have_nvidia = 1 ] && { add_lane dc_gpu_acc "$fc"; add_lane dc_gpu_omp "$fc"; };;
      ifx)       [ $have_intel  = 1 ] && add_lane dc_gpu_vendor "$fc";;
      amdflang|amdflang-new|flang|flang-new)
                 [ $have_amd    = 1 ] && add_lane dc_gpu_vendor "$fc";;
   esac
done
# The head-to-head lane: needs nvfortran (host_data bridge) AND nvcc AND a
# device. It emits the dc_gpu_acc / cuda_faithful / cuda_opt rows together.
# No HIP equivalent yet: it would need hipcc-built kernels linked by an AMD
# Fortran compiler that can do `host_data use_device`, which is not wired.
if [ $have_nvcc = 1 ] && [ $have_nvidia = 1 ]; then
   for fc in "${FCS[@]}"; do
      case "${fc##*/}" in
         nvfortran) [ "${DCCAP[$fc]}" = 1 ] && add_lane cmp "$fc";;
      esac
   done
fi
[ $have_hipcc = 1 ] && [ $have_amd = 1 ] && \
   echo "  (note: hipcc+AMD present but there is no Fortran-launched HIP cmp lane)" >&2

# `MODES=dc` -- every `do concurrent` lane and nothing else. On a non-NVIDIA box
# the cmp lane is not offered anyway (it needs nvfortran + nvcc + an NVIDIA
# device), but this makes the intent explicit and also drops cmp on a machine
# that HAS all three.
case "$MODES" in
   dc|dconly)
      MODES="serial_do dc_serial dc_multicore dc_gpu_acc dc_gpu_omp dc_gpu_vendor";;
esac

if [ "$MODES" != auto ]; then
   keep=()
   for l in "${LANES[@]}"; do
      case " $MODES " in *" ${l%%:*} "*) keep+=("$l");; esac
   done
   if [ ${#keep[@]} -gt 0 ]; then LANES=("${keep[@]}"); else LANES=(); fi
fi

# ---------------------------------------------------------------------------
# Per-measurement configuration (globals, set by the experiment loops)
# ---------------------------------------------------------------------------
cfg_exp=""; cfg_mode=""; cfg_fc=""
cfg_nz=30; cfg_nzstack=128; cfg_bcopy=0
cfg_nxp=473; cfg_nyp=297; cfg_land=0
cfg_threads=1; cfg_bind=close
cfg_variant=faithful; cfg_vlen=1
cfg_cuvariant=faithful; cfg_fullcopy=0; cfg_optnzmax=48; cfg_minblocks=0
cfg_stackpol=prod; cfg_copypol=prod
cfg_cnt=0
cfg_tripcount=host

# Fill MKVARS/MKTARGET for the current cfg_*.
set_make_args() {
   local cls; cls="$(mode_class "$cfg_mode")"
   case "$cls" in
      cmp)
         # BCOPY and FULLCOPY are set from the SAME copy policy and land in the
         # SAME binary, so the 1:1 pairing cannot be got wrong by accident.
         MKTARGET=cmp
         MKVARS=(FC="$cfg_fc" NZSTACK="$cfg_nzstack" VARIANT="$cfg_variant"
                 VLEN="$cfg_vlen" BCOPY="$cfg_bcopy" CNT="$cfg_cnt"
                 FULLCOPY="$cfg_fullcopy" OPT_NZMAX="$cfg_optnzmax"
                 MINBLOCKS="$cfg_minblocks")
         ;;
      serial_do)
         MKTARGET=serialdo
         MKVARS=(FC="$cfg_fc" NZSTACK="$cfg_nzstack" VARIANT="$cfg_variant"
                 VLEN="$cfg_vlen" BCOPY="$cfg_bcopy" CNT="$cfg_cnt")
         ;;
      serial)
         MKTARGET=dc
         MKVARS=(FC="$cfg_fc" DATA=none DC_HOST_FLAGS= NZSTACK="$cfg_nzstack"
                 VARIANT="$cfg_variant" VLEN="$cfg_vlen" BCOPY="$cfg_bcopy"
                 CNT="$cfg_cnt")
         ;;
      multicore)
         local hf; hf="$(mkprint_host DC_HOST_FLAGS "$cfg_fc")"
         MKTARGET=dc
         MKVARS=(FC="$cfg_fc" DATA=none "DC_HOST_FLAGS=$hf" NZSTACK="$cfg_nzstack"
                 VARIANT="$cfg_variant" VLEN="$cfg_vlen" BCOPY="$cfg_bcopy"
                 CNT="$cfg_cnt")
         ;;
      gpu)
         MKTARGET=dc
         case "$cfg_mode" in
            dc_gpu_acc) MKVARS=(FC="$cfg_fc" DATA=acc);;
            dc_gpu_omp) MKVARS=(FC="$cfg_fc" DATA=omp);;
            dc_gpu_vendor)
               local gf; gf="$(mkprint_host DC_GPU_FLAGS "$cfg_fc")"
               MKVARS=(FC="$cfg_fc" DATA=omp "DC_MODE_FLAGS=$gf");;
         esac
         MKVARS+=(NZSTACK="$cfg_nzstack" VARIANT="$cfg_variant" VLEN="$cfg_vlen"
                  BCOPY="$cfg_bcopy" CNT="$cfg_cnt")
         ;;
   esac
   case "$(mode_class "$cfg_mode")" in
      gpu|cmp) MKVARS+=(TRIPCOUNT="$cfg_tripcount");;
   esac
   # Always last, so the detected architecture reaches both the Fortran and the
   # CUDA compile lines of whichever target was selected.
   [ ${#ARCHVARS[@]} -gt 0 ] && MKVARS+=("${ARCHVARS[@]}")
}

binvar_for() {
   case "$(mode_class "$1")" in
      cmp)       echo CMPBIN;;
      serial_do) echo SDBIN;;
      *)         echo DCBIN;;
   esac
}
bldvar_for() {
   case "$(mode_class "$1")" in
      cmp)       echo CMPBLD;;
      serial_do) echo SDBLD;;
      *)         echo DCBLD;;
   esac
}

# ---------------------------------------------------------------------------
# Build with caching. The build dir name encodes the compile-time axes but NOT
# the compiler, so a stamp file catches "same config, different FC" and forces
# a wipe -- otherwise ifx would happily link nvfortran's .mod files.
# ---------------------------------------------------------------------------
BIN=""; BLD=""; BUILD_LOG=""
build_cfg() {
   set_make_args
   BIN="$(mkprint "$(binvar_for "$cfg_mode")")"
   BLD="$(mkprint "$(bldvar_for "$cfg_mode")")"
   [ -n "$BIN" ] && [ -n "$BLD" ] || { BUILD_LOG="could not resolve binary name from make"; return 1; }

   local stamp="$KDIR/$BLD/.harness_stamp"
   local want="${cfg_fc}|${MKVARS[*]}"
   if [ -f "$stamp" ] && [ "$(cat "$stamp" 2>/dev/null)" != "$want" ]; then
      rm -rf "$KDIR/$BLD" "$KDIR/$BIN"
   fi
   local out
   if ! out=$( cd "$KDIR" && make "$MKTARGET" "${MKVARS[@]}" 2>&1 ); then
      BUILD_LOG="$out"
      return 1
   fi
   mkdir -p "$KDIR/$BLD" && printf '%s\n' "$want" > "$stamp"
   return 0
}

# ---------------------------------------------------------------------------
# Run one invocation. Echoes the driver's stdout; sets RUN_RC.
# ---------------------------------------------------------------------------
RUN_RC=0
run_bin() {   # $1 = reps (0 = auto), $2 = warm
   local reps="$1" warm="$2"
   # ONE argument order, because every driver is now the same Fortran driver.
   local -a a=("$cfg_nxp" "$cfg_nyp" "$cfg_nz" "$reps" "$warm" "$cfg_land")
   # Thread-stack budget. ks_solve_column holds ~60 private column arrays of
   # (NZSTACK+1) x VLEN doubles, and ks_projected_state / ks_adaptive_dt add
   # another ~10 on top of that frame. Round the estimate UP (integer division
   # was truncating 1.9 MB to 1) and give it 8x headroom, floor 64 MB -- a
   # too-small thread stack here is a segfault that reads like a code bug.
   local frame_mb=$(( (60 * (cfg_nzstack + 1) * cfg_vlen * 8 + 1048575) / 1048576 ))
   local stk=$(( frame_mb * 8 )); [ "$stk" -lt 64 ] && stk=64
   # THREAD COUNT IS NOT PORTABLE ACROSS COMPILERS AND THE DIFFERENCE IS SILENT.
   # nvfortran's `-stdpar=multicore` ignores OMP_NUM_THREADS completely and reads
   # ACC_NUM_CORES; ifx (-qopenmp) and flang/amdflang (-fopenmp) read
   # OMP_NUM_THREADS and ignore ACC_NUM_CORES. Measured on nvfortran 26.5:
   # OMP_NUM_THREADS=1 -> 3.91 ms (no effect), ACC_NUM_CORES=1 -> 78.81 ms.
   # Setting only one of them yields a thread-scaling curve that is flat at
   # 1.00x because every point silently ran on all cores. Set BOTH; whichever
   # the compiler does not use is inert.
   local -a envv=(KS_TARGET_MS="$KS_TARGET_MS" KS_MAX_REPS="$KS_MAX_REPS")
   case "$(mode_class "$cfg_mode")" in
      gpu|cmp)
         # Device-resident: the host thread count is irrelevant here.
         ;;
      *)
         envv+=(OMP_NUM_THREADS="$cfg_threads" ACC_NUM_CORES="$cfg_threads"
                OMP_PROC_BIND="$cfg_bind" OMP_PLACES=cores
                OMP_STACKSIZE="${stk}M")
         ;;
   esac
   local out
   if [ "$(mode_locks "$cfg_mode")" = 1 ]; then
      out=$( cd "$KDIR" && env "${envv[@]}" "$LOCK" "ks-$cfg_mode" \
             timeout -k 5 "$RUN_TIMEOUT" ./"$BIN" "${a[@]}" 2>&1 )
   else
      out=$( cd "$KDIR" && env "${envv[@]}" \
             timeout -k 5 "$RUN_TIMEOUT" ./"$BIN" "${a[@]}" 2>&1 )
   fi
   RUN_RC=$?
   printf '%s\n' "$out"
}

# Parse ONE RESULT line into the R associative array. A cmp run emits several
# (one per implementation), so the caller iterates.
declare -A R
parse_line() {
   local tok
   R=()
   for tok in $1; do
      case "$tok" in *=*) R[${tok%%=*}]="${tok#*=}";; esac
   done
   [ -n "${R[ms]:-}" ]
}

# Registers + per-thread frame for the `do concurrent` column kernel, read out
# of the built binary. Prints "REG STACK" (empty if unavailable).
#
# Picks the nvfortran-generated entry by name (`..._column_kernel_<line>_gpu`)
# rather than by size: a cmp binary also contains the two nvcc kernels, and
# "biggest frame wins" would sometimes select one of those instead.
gpu_res_usage() {
   [ -n "$CUOBJDUMP" ] && [ -f "$KDIR/$1" ] || { echo ""; return; }
   "$CUOBJDUMP" -res-usage "$KDIR/$1" 2>/dev/null | awk '
      /^ Function /{ f=$2; sub(/:$/,"",f) }
      /REG:/{
         r=0; st=0
         if (match($0,/REG:[0-9]+/))   r  = substr($0,RSTART+4,RLENGTH-4)+0
         if (match($0,/STACK:[0-9]+/)) st = substr($0,RSTART+6,RLENGTH-6)+0
         if (f ~ /column_kernel_[0-9]+_gpu/) { br=r; bs=st }
      }
      END{ if (br+0>0) print br, bs }'
}

median() {  # numbers on stdin
   sort -g | awk '{v[NR]=$1} END{ if(NR==0){print ""; exit} print (NR%2)? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2 }'
}

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
CSV="$DATADIR/ks_sweep_${HOSTN}_${TS}.csv"
KS_COLS="experiment,mode,impl,launcher,variant,lanes,compiler,compiler_ver,flags,\
stack_policy,copy_policy,nzstack,bcopy,opt_nzmax,minblocks,\
threads,proc_bind,device,gpu_arch,nxp,nyp,nx,ny,ncols,nwet,nz,land_pct,\
reps,warm,nrun,ms_min,ms_med,ns_per_col,it_outer,it_inner,\
kd_sum,kd_min,kd_max,regs,local_b,status,git_hash,host,timestamp"
NROWS=0

# emit ONE row for one implementation out of this configuration's run.
#   ks_emit <impl> <status> <ms_min> <ms_med> <ns_per_col> <it_outer> <it_inner>
# `impl` is dc | cuda_faithful | cuda_opt. Fields not carried by the RESULT line
# come from the cfg_* globals.
ks_emit() {
   local impl="$1" dev ver flags cpol launcher
   dev="$(device_for "$(mode_for_impl "$impl")")"
   case "$1" in
      cuda_*) ver="$( "$NVCC_BIN" --version 2>/dev/null | tail -1 )"
              flags="$(mkprint CMP_CUFLAGS)";;
      *)      ver="$(fc_version "$cfg_fc")"
              case "$(mode_class "$cfg_mode")" in
                 cmp) flags="$(mkprint CMP_FFLAGS)";;
                 *)   flags="$(mkprint BASE_FFLAGS) $(mkprint DC_MODE_FLAGS)";;
              esac;;
   esac
   # cuda_opt ALWAYS bounds the whole-array copy -- it cannot express the `prod`
   # policy at all, so label it honestly or a reader pairing rows by copy_policy
   # will mispair it against a DC BCOPY=0 row.
   cpol="$cfg_copypol"
   [ "$impl" = cuda_opt ] && cpol=opt
   case "$(mode_class "$cfg_mode")" in cmp) launcher=cmp;; *) launcher=solo;; esac
   printf '%s,%s,%s,%s,%s,%s,%s,"%s","%s",%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$cfg_exp" "$(mode_for_impl "$impl")" "$impl" "$launcher" \
      "${R[variant]:-$cfg_variant}" "${R[lanes]:-$cfg_vlen}" \
      "$cfg_fc" "$(_q "$ver")" "$(_q "$flags")" \
      "$cfg_stackpol" "$cpol" "${R[nzstack]:-$cfg_nzstack}" "${R[bcopy]:-$cfg_bcopy}" \
      "$cfg_optnzmax" "$cfg_minblocks" \
      "$cfg_threads" "$cfg_bind" "$(_q "$dev")" "${CFG_ARCH:-default}" \
      "$cfg_nxp" "$cfg_nyp" "${R[nx]:-}" "${R[ny]:-}" "${R[ncols]:-}" "${R[nwet]:-}" \
      "$cfg_nz" "$cfg_land" "${R[reps]:-}" "$KS_WARM" "$NRUN" \
      "$3" "$4" "$5" "$6" "$7" \
      "${R[kd_sum]:-}" "${R[kd_min]:-}" "${R[kd_max]:-}" \
      "${R[regs]:-}" "${R[local_b]:-}" "$2" "$GHASH" "$HOSTN" "$TS" >> "$CSV"
   NROWS=$((NROWS+1))
}

# ---------------------------------------------------------------------------
# measure(): build, NRUN timed invocations (min + median), optional counters
# pass, one CSV row. Every failure path still writes a row -- a sweep whose
# gaps are invisible is worse than one that says NaN and why.
# ---------------------------------------------------------------------------
declare -A MSL RLINE
measure() {
   local label
   label=$(printf '%-12s %-9s nz=%-4s stk=%-4s bc=%s %sx%s thr=%-3s %s' \
      "$cfg_mode" "$cfg_fc" "$cfg_nz" "$cfg_nzstack" "$cfg_bcopy" \
      "$cfg_nxp" "$cfg_nyp" "$cfg_threads" \
      "$( [ "$cfg_variant" = block ] && echo "vlen=$cfg_vlen" || echo "" )")

   if [ "$DRY" = 1 ]; then printf '    %s\n' "$label"; NROWS=$((NROWS+1)); return; fi

   cfg_cnt=0
   if ! build_cfg; then
      printf '    %s  BUILD_FAIL\n' "$label"
      printf '%s\n' "$BUILD_LOG" > "$KDIR/.ks_sweep_build.log"
      R=(); ks_emit dc BUILD_FAIL NaN NaN NaN NaN NaN
      return
   fi

   # A cmp run emits one RESULT line per implementation; a dc/serialdo run emits
   # one. Accumulate ms per impl across the NRUN invocations, keep the last line
   # of each for its non-timing fields, and preserve first-seen order.
   MSL=(); RLINE=(); local order="" out rc i line imp msv
   for i in $(seq 1 "$NRUN"); do
      out=$(run_bin 0 "$KS_WARM"); rc=$RUN_RC
      if [ "$rc" -ne 0 ]; then
         printf '%s\n' "$out" > "$KDIR/.ks_sweep_run.log"
         if [ "$rc" -ge 124 ]; then
            printf '    %s  TIMEOUT (%ss)\n' "$label" "$RUN_TIMEOUT"
            R=(); ks_emit dc TIMEOUT NaN NaN NaN NaN NaN; return
         fi
         printf '    %s  RUN_FAIL rc=%s\n' "$label" "$rc"
         R=(); ks_emit dc RUN_FAIL NaN NaN NaN NaN NaN; return
      fi
      while IFS= read -r line; do
         parse_line "$line" || continue
         imp="${R[impl]:-dc}"; msv="${R[ms]}"
         case " $order " in *" $imp "*) ;; *) order="$order $imp";; esac
         MSL[$imp]="${MSL[$imp]:-} $msv"
         RLINE[$imp]="$line"
      done < <(printf '%s\n' "$out" | grep '^RESULT ')
   done
   if [ -z "$order" ]; then
      printf '    %s  NO_RESULT_LINE\n' "$label"
      R=(); ks_emit dc NO_RESULT NaN NaN NaN NaN NaN; return
   fi

   # Register/frame usage of the DC kernel, straight out of the binary.
   DC_REGS=""; DC_STACK=""
   case "$(mode_class "$cfg_mode")" in
      gpu|cmp) read -r DC_REGS DC_STACK <<<"$(gpu_res_usage "$BIN")";;
   esac

   # Work metric for the Fortran side. The CUDA kernels count unconditionally;
   # `do concurrent` needs a -DKS_COUNTERS build, which is NEVER timed (it adds
   # registers to a register-bound kernel), so it runs separately at one rep.
   local dc_ito="" dc_iti=""
   if [ "$COUNTERS_PASS" = 1 ]; then
      local save_bin="$BIN" save_bld="$BLD"
      cfg_cnt=1
      if build_cfg; then
         local cout
         cout=$(run_bin 1 0)
         if [ $RUN_RC -eq 0 ]; then
            while IFS= read -r line; do
               parse_line "$line" || continue
               if [ "${R[impl]:-dc}" = dc ]; then dc_ito="${R[it_outer]}"; dc_iti="${R[it_inner]}"; fi
            done < <(printf '%s\n' "$cout" | grep '^RESULT ')
         fi
      fi
      cfg_cnt=0
      set_make_args               # ks_emit must report the TIMED build's flags
      BIN="$save_bin"; BLD="$save_bld"
   fi

   local imp ms_min ms_med nspc ito iti
   for imp in $order; do
      ms_min=$(printf '%s\n' ${MSL[$imp]} | sort -g | head -1)
      ms_med=$(printf '%s\n' ${MSL[$imp]} | median)
      parse_line "${RLINE[$imp]}" || continue
      nspc=$(awk -v m="$ms_min" -v n="${R[ncols]:-1}" 'BEGIN{printf "%.6f", m*1e6/n}')
      ito="${R[it_outer]:-}"; iti="${R[it_inner]:-}"
      if [ "$imp" = dc ] && [ -n "$dc_iti" ]; then ito="$dc_ito"; iti="$dc_iti"; fi
      # The Fortran driver cannot report its own kernel's registers; cuobjdump can.
      if [ "$imp" = dc ] && [ -n "$DC_REGS" ]; then
         R[regs]="$DC_REGS"; R[local_b]="$DC_STACK"
      fi
      printf '    %s  %-14s %12s ms  %11s ns/col  it=%s\n' \
         "$label" "$imp" "$ms_min" "$nspc" "${iti:-?}"
      ks_emit "$imp" OK "$ms_min" "$ms_med" "$nspc" "$ito" "$iti"
   done
}

# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------
# stack policy -> cfg_nzstack / cfg_optnzmax. Returns 1 if the policy cannot
# express this nz (prod cannot hold nz+1 > 128).
apply_stack_policy() {
   cfg_stackpol="$1"
   case "$1" in
      prod) [ $((cfg_nz + 1)) -le 128 ] || return 1
            cfg_nzstack=128; cfg_optnzmax=128;;
      fit)  cfg_nzstack=$((cfg_nz + 1)); cfg_optnzmax=$((cfg_nz + 1));;
      *) return 1;;
   esac
   return 0
}

# copy policy -> the MATCHED pair of DC BCOPY and CUDA FULLCOPY. Mixing these
# times the whole-array copy instead of the codegen, so they are set together
# and never independently.
apply_copy_policy() {
   cfg_copypol="$1"
   case "$1" in
      prod) cfg_bcopy=0; cfg_fullcopy=1;;   # both O(NZSTACK) -- true 1:1
      opt)  cfg_bcopy=1; cfg_fullcopy=0;;   # both O(nz)
      *) return 1;;
   esac
   return 0
}

set_cols() { cfg_nxp="${1%x*}"; cfg_nyp="${1#*x}"; }

# columns to use for a mode: GPU lanes need the production width to be
# occupancy-honest; serial CPU lanes would take hours there.
cols_for_mode() {
   case "$(mode_class "$1")" in
      gpu|cmp)   echo "$COLS_GPU";;
      *)         echo "$COLS_CPU";;
   esac
}
too_wide_for_serial() {
   case "$(mode_class "$cfg_mode")" in serial|serial_do) ;; *) return 1;; esac
   local n=$(( (cfg_nxp + 6) * (cfg_nyp + 6) ))
   [ "$n" -gt "$SERIAL_MAX_COLS" ]
}

# For gpuc lanes the "compiler" is the GPU-C compiler and cuvariant is fixed.
set_lane() {
   cfg_mode="${1%%:*}"; cfg_fc="${1##*:}"
   # The cmp binary carries BOTH hand-written kernels, so there is no per-lane
   # CUDA variant to choose any more; the per-impl labelling happens in ks_emit.
   case "$cfg_mode" in
      cmp) cfg_cuvariant=both;;
      *)   cfg_cuvariant=na;;
   esac
}

thread_list() {
   if [ "$THREAD_LIST" != auto ]; then echo "$THREAD_LIST"; return; fi
   local t=1 out=""
   while [ "$t" -lt "$NCORES" ]; do out="$out $t"; t=$((t*2)); done
   echo "$out $NCORES"
}

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
exp_nz() {
   cfg_exp=nz; cfg_variant=faithful; cfg_vlen=1; cfg_land=$LAND_PCT
   cfg_minblocks=0
   [ ${#LANES[@]} -gt 0 ] || { echo "  (no runnable lanes on this host)"; return; }
   for lane in "${LANES[@]}"; do
      apply_copy_policy prod        # the 1:1 baseline pairing, re-applied per
      set_lane "$lane"              # lane because set_lane may override it
      cfg_threads=1
      [ "$(mode_class "$cfg_mode")" = multicore ] && cfg_threads=$NCORES
      set_cols "$(cols_for_mode "$cfg_mode")"
      echo "  --- nz sweep: $cfg_mode ($cfg_fc) at ${cfg_nxp}x${cfg_nyp} ---"
      for pol in $STACK_POLICIES; do
         for nz in $NZ_LIST; do
            cfg_nz=$nz
            apply_stack_policy "$pol" || { printf '    skip nz=%-4s stack=%s (nz+1 > 128)\n' "$nz" "$pol"; continue; }
            too_wide_for_serial && { printf '    skip nz=%-4s (serial lane, %s cols > SERIAL_MAX_COLS=%s)\n' "$nz" "$(( (cfg_nxp+6)*(cfg_nyp+6) ))" "$SERIAL_MAX_COLS"; continue; }
            measure
         done
      done
   done
}

exp_copy() {
   cfg_exp=copy; cfg_variant=faithful; cfg_vlen=1; cfg_land=$LAND_PCT
   cfg_minblocks=0
   [ ${#LANES[@]} -gt 0 ] || { echo "  (no runnable lanes on this host)"; return; }
   for lane in "${LANES[@]}"; do
      set_lane "$lane"
      cfg_threads=1
      [ "$(mode_class "$cfg_mode")" = multicore ] && cfg_threads=$NCORES
      set_cols "$(cols_for_mode "$cfg_mode")"
      echo "  --- copy policy: $cfg_mode ($cfg_fc) at ${cfg_nxp}x${cfg_nyp} ---"
      for nz in $NZ_REF $NZ_DEEP; do
         cfg_nz=$nz
         for pol in $STACK_POLICIES; do
            apply_stack_policy "$pol" || continue
            too_wide_for_serial && continue
            for cp in $COPY_POLICIES; do
               apply_copy_policy "$cp"
               measure
            done
         done
      done
   done
}

exp_cols() {
   cfg_exp=cols; cfg_variant=faithful; cfg_vlen=1; cfg_land=$LAND_PCT
   cfg_minblocks=0; cfg_nz=$NZ_REF
   apply_stack_policy prod || return
   [ ${#LANES[@]} -gt 0 ] || { echo "  (no runnable lanes on this host)"; return; }
   for lane in "${LANES[@]}"; do
      apply_copy_policy prod        # per lane: set_lane may override it
      set_lane "$lane"
      cfg_threads=1
      [ "$(mode_class "$cfg_mode")" = multicore ] && cfg_threads=$NCORES
      echo "  --- width sweep: $cfg_mode ($cfg_fc) at nz=$cfg_nz ---"
      for c in $COLS_LIST; do
         set_cols "$c"
         too_wide_for_serial && { printf '    skip %-9s (serial lane > SERIAL_MAX_COLS)\n' "$c"; continue; }
         measure
      done
   done
}

exp_threads() {
   cfg_exp=threads; cfg_variant=faithful; cfg_vlen=1; cfg_land=$LAND_PCT
   cfg_minblocks=0; cfg_nz=$NZ_REF
   apply_stack_policy prod || return
   set_cols "$COLS_CPU"
   local any=0
   [ ${#LANES[@]} -gt 0 ] || { echo "  (no runnable lanes on this host)"; return; }
   for lane in "${LANES[@]}"; do
      apply_copy_policy prod        # per lane: set_lane may override it
      set_lane "$lane"
      [ "$(mode_class "$cfg_mode")" = multicore ] || continue
      any=1
      echo "  --- thread scaling: $cfg_mode ($cfg_fc) nz=$cfg_nz at ${cfg_nxp}x${cfg_nyp} ---"
      for b in $BIND_LIST; do
         cfg_bind="$b"
         for t in $(thread_list); do
            cfg_threads="$t"
            measure
         done
      done
   done
   cfg_bind=close; cfg_threads=1
   [ "$any" = 1 ] || echo "  (skip threads: no dc_multicore lane on this host)"
}

exp_vlen() {
   if [ ! -f "$KDIR/ks_block.F90" ]; then
      echo "  (skip vlen: kappa_shear/ks_block.F90 not present yet)"
      return
   fi
   cfg_exp=vlen; cfg_variant=block; cfg_land=$LAND_PCT; cfg_minblocks=0
   [ ${#LANES[@]} -gt 0 ] || { echo "  (no runnable lanes on this host)"; return; }
   for lane in "${LANES[@]}"; do
      apply_copy_policy opt         # the blocked kernel is the optimised lane
      set_lane "$lane"
      [ "$(mode_class "$cfg_mode")" = cmp ] && continue    # no blocked CUDA kernel
      cfg_threads=1
      [ "$(mode_class "$cfg_mode")" = multicore ] && cfg_threads=$NCORES
      set_cols "$(cols_for_mode "$cfg_mode")"
      echo "  --- VLEN sweep: $cfg_mode ($cfg_fc) at ${cfg_nxp}x${cfg_nyp} ---"
      for nz in $NZ_REF $NZ_DEEP; do
         cfg_nz=$nz
         apply_stack_policy fit || continue
         too_wide_for_serial && continue
         # The faithful per-column arrangement, measured HERE rather than
         # borrowed from the nz experiment: same binary flags, same stack
         # policy, same machine state. Comparing blocking against a baseline
         # taken under different conditions is how you get a ratio nobody can
         # defend.
         cfg_variant=faithful; cfg_vlen=1
         measure
         cfg_variant=block
         for v in $VLEN_LIST; do
            cfg_vlen="$v"
            measure
         done
      done
   done
   cfg_variant=faithful; cfg_vlen=1
}

exp_cuopt() {
   cfg_exp=cuopt; cfg_variant=faithful; cfg_vlen=1; cfg_land=$LAND_PCT
   [ ${#LANES[@]} -gt 0 ] || { echo "  (no runnable lanes on this host)"; return; }
   for lane in "${LANES[@]}"; do
      apply_copy_policy opt
      set_lane "$lane"
      [ "$(mode_class "$cfg_mode")" = cmp ] || continue
      cfg_threads=1
      set_cols "$COLS_GPU"
      echo "  --- CUDA opt knobs: $cfg_mode ($cfg_fc) at ${cfg_nxp}x${cfg_nyp} ---"
      for nz in $NZ_REF $NZ_DEEP; do
         cfg_nz=$nz
         apply_stack_policy fit || continue
         # MINBLOCKS only affects opt_kernel.cu, but it is a build flag of the
         # whole cmp binary, so each value is a separate build+run whose
         # cuda_faithful/dc rows should be identical -- a free consistency check.
         for mb in $MINBLOCKS_LIST; do cfg_minblocks="$mb"; measure; done
         cfg_minblocks=0
      done
   done
}

exp_tripcount() {
   # Is -gpu=tripcount:host still load-bearing on THIS machine + compiler?
   # On NVHPC 26.5 / V100 its absence makes the DC column solve ~2x slower,
   # which is indistinguishable from "CUDA wins big here" unless you check.
   # Runs across the depth range because the effect is on a multi-loop region
   # whose trip counts scale with nz.
   cfg_exp=tripcount; cfg_variant=faithful; cfg_vlen=1; cfg_land=$LAND_PCT
   cfg_minblocks=0
   [ ${#LANES[@]} -gt 0 ] || { echo "  (no runnable lanes on this host)"; return; }
   for lane in "${LANES[@]}"; do
      apply_copy_policy prod
      set_lane "$lane"
      case "$(mode_class "$cfg_mode")" in gpu|cmp) ;; *) continue;; esac
      cfg_threads=1
      set_cols "$COLS_GPU"
      echo "  --- tripcount A/B: $cfg_mode ($cfg_fc) at ${cfg_nxp}x${cfg_nyp} ---"
      for nz in $NZ_LIST; do
         cfg_nz=$nz
         apply_stack_policy prod || continue
         for tc in host off; do
            cfg_tripcount="$tc"
            measure
         done
      done
      cfg_tripcount=host
   done
}

# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------
echo "=================================================================="
echo "  kappa_shear sweep -- experiments: ${WANT[*]}"
echo "  host      : $HOSTN   cores=$NCORES   git=$GHASH"
echo "  compilers : ${FCS[*]}"
echo "  devices   : NVIDIA=$have_nvidia AMD=$have_amd Intel=$have_intel  nvcc=$have_nvcc hipcc=$have_hipcc"
echo "  gpu arch  : ${CFG_ARCH:-<none detected -- Makefile default>} / ${CFG_NVARCH:-<default>}  (compute_cap ${GPU_CC:-n/a})"
[ $have_amd = 1 ] && echo "  amd arch  : $( cd "$KDIR" && make -s print-AMD_GPU_ARCH 2>/dev/null )  (from rocminfo)"
echo "  lanes     : ${LANES[*]:-<none>}"
echo "  discipline: NRUN=$NRUN target=${KS_TARGET_MS}ms reps<=$KS_MAX_REPS warm=$KS_WARM timeout=${RUN_TIMEOUT}s counters=$COUNTERS_PASS"
[ "$DRY" = 1 ] && echo "  *** DRY RUN -- nothing is built or executed ***"
echo "=================================================================="

if [ "$DRY" != 1 ]; then
   case " ${LANES[*]} " in
      *dc_gpu*|*cuda_*|*hip_*)
         ensure_lock
         gpu_assert_free || { echo "Refusing to start: see NOTES_ON_PERF.md on exclusive-GPU discipline." >&2; exit 1; };;
   esac
   printf '%s\n' "$KS_COLS" > "$CSV"
   echo "  -> $CSV"; echo
fi

for e in "${WANT[@]}"; do
   echo "=== experiment: $e ==============================================="
   "exp_$e"
   echo
done

echo "=================================================================="
if [ "$DRY" = 1 ]; then
   echo "  dry run: $NROWS configurations would be measured"
   echo "  rough cost: ~$(( NROWS * (NRUN * KS_TARGET_MS / 1000 + 2) ))s of RUN time, plus builds"
else
   echo "  done -> $CSV"
   echo "  rows: $NROWS   ($(grep -c ',OK,' "$CSV" || true) OK)"
   echo "  next: tools/ks_report.py $CSV"
fi
echo "=================================================================="
