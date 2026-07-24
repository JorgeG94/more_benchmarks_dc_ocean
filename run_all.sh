#!/usr/bin/env bash
# run_all.sh -- the paper's PART 1 matrix in one command: every runnable
# {mode x compiler} combination on THIS machine, auto-detected. One combined
# provenance CSV; whatever this box can't do is SKIPPED with a printed reason,
# so results from many machines aggregate by (mode,compiler,device).
#
# Modes (the baselines):
#   serial_do      plain nested `do` (committed serialdo/); ANY F2008 compiler
#   dc_serial      `do concurrent`, no parallel flag                (DC-capable FC)
#   dc_multicore   `do concurrent` across CPU cores                 (DC-capable FC + a host flag)
#   dc_gpu_acc     `do concurrent` on GPU, OpenACC data layer       (nvfortran + NVIDIA GPU)
#   dc_gpu_omp     `do concurrent` on GPU, OpenMP-target data layer (nvfortran NVIDIA; ifx Intel; amdflang AMD)
#
# The per-compiler multicore / GPU flags live in config.mk (DC_HOST_FLAGS,
# DC_GPU_FLAGS) -- the one reproducible place -- and are read here via
# `make print-`. Vendor GPU builds override DC_MODE_FLAGS on a DATA=omp build;
# NO kernel Makefile is touched.
#
#   ./run_all.sh [FC ...]     compilers to try (default: auto-detect on PATH)
#   size + kernel list from benchmark_config.mk / benchmark_common.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/benchmark_common.sh"

# ---- which compilers to consider ------------------------------------------
if [ $# -gt 0 ]; then CANDIDATES=("$@"); else CANDIDATES=(nvfortran ifx flang flang-new amdflang gfortran); fi
COMPILERS=()
for fc in "${CANDIDATES[@]}"; do command -v "$fc" >/dev/null 2>&1 && COMPILERS+=("$fc"); done
[ ${#COMPILERS[@]} -gt 0 ] || { echo "no candidate compilers on PATH: ${CANDIDATES[*]}" >&2; exit 1; }

# ---- accelerator detection (structural for Intel/AMD -- unvalidated here) ---
have_nvidia=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && have_nvidia=1
have_intel=0;  command -v sycl-ls    >/dev/null 2>&1 && have_intel=1
have_amd=0;    command -v rocminfo   >/dev/null 2>&1 && have_amd=1

# ---- probe: can THIS compiler compile `do concurrent ... local()`? ---------
# (gfortran <= 14 cannot; this is why gfortran's only lane is serial_do.)
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

ncores=$NCORES

csv="$DATADIR/matrix_${HOSTN}_${TS}.csv"; csv_new "$csv"

banner "PART-1 MATRIX -- all runnable modes x compilers on this host"
echo "  compilers: ${COMPILERS[*]}"
echo "  devices  : NVIDIA=$([ $have_nvidia = 1 ] && echo yes || echo no)  Intel=$([ $have_intel = 1 ] && echo yes || echo no)  AMD=$([ $have_amd = 1 ] && echo yes || echo no)   cores=$ncores"
echo "  -> $csv"; echo

# make-var introspection (print-% lives in common/serialdo.mk, included everywhere)
pflags() { ( cd "$REPO/redi" && make -s "print-$1" "${@:2}" 2>/dev/null ); }

# ---- run one (mode,compiler): build+run every kernel, append rows -----------
# usage: run_mode <label> <bin> <threads> <flags-for-csv> <lock:0|1> <fc> <ver> \
#                 <builddir-to-wipe> -- <make args...>
run_mode() {
   local label="$1" bin="$2" threads="$3" flags="$4" lock="$5" fc="$6" ver="$7" bdir="$8"; shift 8
   [ "$1" = "--" ] && shift
   local margs=("$@") dev; dev="$(device_for "$label")"
   printf '  === %-14s FC=%-10s threads=%-4s ===\n' "$label" "$fc" "$threads"
   local k out ms bstate
   for k in "${ALL_KERNELS[@]}"; do
      rm -rf "$REPO/$k/$bdir"                     # fresh build: never reuse another compiler's objects
      if out=$( cd "$REPO/$k" && make "${margs[@]}" 2>&1 ); then
         bstate="built"
      else
         ( cd "$REPO/$k" && echo "$out" > .matrix_build.log )
         printf '     %-22s %-11s FAILED (log: %s/.matrix_build.log)\n' "$k" "-" "$k"
         emit_row "$csv" "$k" "$label" "$fc" "$ver" "$flags" "$threads" "$dev" NaN
         continue
      fi
      if [ "$lock" = 1 ]; then
         out=$( cd "$REPO/$k" && OMP_NUM_THREADS=$threads OMP_PROC_BIND=close "$LOCK" "$k-$label" ./"$bin" $ARGS 2>&1 )
      else
         out=$( cd "$REPO/$k" && OMP_NUM_THREADS=$threads OMP_PROC_BIND=close ./"$bin" $ARGS 2>&1 )
      fi
      ms=$(parse_ms "$out")
      printf '     %-22s %-11s %14s\n' "$k" "$bstate" "${ms:-**FAIL**}"
      emit_row "$csv" "$k" "$label" "$fc" "$ver" "$flags" "$threads" "$dev" "${ms:-NaN}"
   done
   echo
}

for fc in "${COMPILERS[@]}"; do
   ver=$("$fc" --version 2>&1 | head -1 | tr ',' ' ')
   bf=$(pflags BASE_FFLAGS FC=$fc)
   dccap=0; dc_capable "$fc" && dccap=1
   echo "------------------------------------------------------------------"
   echo "  $fc : $ver   [do concurrent: $([ $dccap = 1 ] && echo YES || echo 'NO -> serial_do only')]"
   echo "------------------------------------------------------------------"

   # 1) serial_do -- always (any F2008 compiler)
   run_mode serial_do sdo 1 "$(pflags SD_FFLAGS FC=$fc)" 0 "$fc" "$ver" build/serialdo \
            -- serialdo FC=$fc

   if [ $dccap = 0 ]; then
      echo "  (skip dc_serial / dc_multicore / dc_gpu: $fc cannot compile do concurrent local())"; echo
      continue
   fi

   # 2) dc_serial -- DC compiled with no parallel flag
   run_mode dc_serial dc_none 1 "$bf [DATA=none, no host flag]" 0 "$fc" "$ver" build/dc_none \
            -- dc DATA=none DC_HOST_FLAGS= FC=$fc

   # 3) dc_multicore -- DC across cores, using config.mk's per-compiler host flag
   mcf=$(pflags DC_HOST_FLAGS FC=$fc)
   if [ -n "$mcf" ]; then
      run_mode dc_multicore dc_none "$ncores" "$bf $mcf" 0 "$fc" "$ver" build/dc_none \
               -- dc DATA=none "DC_HOST_FLAGS=$mcf" FC=$fc
   else
      echo "  (skip dc_multicore: no host-parallel do-concurrent flag in config.mk for $fc)"; echo
   fi

   # 4) dc_gpu -- vendor-specific offload
   case "${fc##*/}" in
      nvfortran)
         if [ $have_nvidia = 1 ]; then
            ensure_lock; gpu_assert_free || { echo "  (skip dc_gpu: NVIDIA GPU not free)"; echo; continue; }
            for layer in acc omp; do
               run_mode "dc_gpu_$layer" "dc_$layer" 1 "$bf $(pflags DC_MODE_FLAGS DATA=$layer FC=$fc)" 1 \
                        "$fc" "$ver" build/dc_$layer -- dc DATA=$layer FC=$fc
            done
         else echo "  (skip dc_gpu: no NVIDIA GPU on this host)"; echo; fi
         ;;
      ifx)
         gf=$(pflags DC_GPU_FLAGS FC=$fc)
         if [ $have_intel = 1 ] && [ -n "$gf" ]; then
            run_mode dc_gpu_omp dc_omp 1 "$bf $gf" 0 "$fc" "$ver" build/dc_omp \
                     -- dc DATA=omp FC=$fc "DC_MODE_FLAGS=$gf"
         else echo "  (skip dc_gpu: no Intel GPU (sycl-ls) on this host)"; echo; fi
         ;;
      amdflang)
         gf=$(pflags DC_GPU_FLAGS FC=$fc)
         if [ $have_amd = 1 ] && [ -n "$gf" ]; then
            run_mode dc_gpu_omp dc_omp 1 "$bf $gf" 0 "$fc" "$ver" build/dc_omp \
                     -- dc DATA=omp FC=$fc "DC_MODE_FLAGS=$gf"
         else echo "  (skip dc_gpu: no AMD GPU (rocminfo) on this host)"; echo; fi
         ;;
      *)
         echo "  (skip dc_gpu: no GPU offload path wired for $fc)"; echo
         ;;
   esac
done

echo "=================================================================="
echo "  done -> $csv"
echo "  rows: $(( $(wc -l < "$csv") - 1 ))"