#!/bin/bash
# Contention-robust measurement harness. EVERY timing number in README.md comes
# through this script.
#
# ⚠ WHY THIS EXISTS. This benchmark shares one V100 with three sibling
# benchmarks (EPBL / Redi / MEKE) driven by other agents. A naive single run
# gave dc = 120.7, 77.5, 135.5, 40.8 ms for the SAME binary -- a 3.3x spread,
# larger than every effect under test. Two conclusions were nearly published off
# that noise: a phantom "NZ_STACK_MAX=32 is 2x slower" and a phantom
# "NZ_STACK_MAX=256 cliff". Both evaporated on repeat. Do not time this kernel
# with a bare `./ks_bench`.
#
# Protocol:
#   1. Wait for an idle GPU (no compute apps AND util < 10%) before each run.
#   2. Run the whole binary N times.
#   3. Report the MINIMUM per variant. Contention only ever ADDS time, so the
#      min is the best estimate of the uncontended cost. med/max are printed
#      too: if min and med diverge materially the run was contended and the
#      number is soft.
#
# ⚠ BASH TRAP, hit twice: `x=$(cmd | grep -c . || echo 0)` yields TWO lines
# ("0\n0") when grep matches nothing, because grep -c prints "0" AND exits 1.
# `[ "$x" -eq 0 ]` then dies with "integer expression expected", wait_idle never
# succeeds, and every run silently reports itself contended. grep -c already
# prints 0 on no match -- no `|| echo` needed.
#
# Usage: ./measure.sh <label> <n_runs> <ks_bench args...>
#   ./measure.sh "0.1deg" 4 473 297 30 20 10 1
set -u
LABEL="$1"; NRUN="$2"; shift 2

wait_idle() {
   local i n u
   for i in $(seq 1 60); do
      n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -z "${u:-}" ] && u=100
      [ -z "${n:-}" ] && n=1
      if [ "$n" -eq 0 ] && [ "$u" -lt 10 ]; then return 0; fi
      sleep 5
   done
   echo "    !! GPU busy 300 s -- CONTENDED, treat the number below as soft"
}

dcs=(); cus=(); nas=()
for r in $(seq 1 "$NRUN"); do
   wait_idle
   out=$(./ks_bench "$@" 2>&1)
   d=$(echo "$out" | grep "do concurrent)" | sed 's/.*: *//;s/ ms.*//')
   c=$(echo "$out" | grep "faithful (1 thread" | sed 's/.*: *//;s/ ms.*//')
   [ -n "$d" ] && dcs+=("$d")
   [ -n "$c" ] && cus+=("$c")

   # The native (cudaMalloc, no OpenACC) driver, when it has been built. Its own
   # wait_idle is NOT redundant: a sibling that starts between the two binaries
   # hits only the second one, which silently turns a CONTENTION artefact into an
   # apparent host_data cost. Seen: native 61.9 and 67.0 ms in a pair whose
   # ks_bench half looked perfectly clean.
   if [ -x ./ks_native ]; then
      wait_idle
      n=$(./ks_native "$@" 2>&1 | grep "CUDA native" | sed 's/.*: *//;s/ ms.*//')
      [ -n "$n" ] && nas+=("$n")
   fi
done

stats() {
   printf '%s\n' "$@" | sort -g | \
      awk '{v[NR]=$1} END{printf "%8.2f %8.2f %8.2f", v[1], v[int((NR+1)/2)], v[NR]}'
}
dmin=$(printf '%s\n' "${dcs[@]}" | sort -g | head -1)
cmin=$(printf '%s\n' "${cus[@]}" | sort -g | head -1)
echo "  $LABEL  [$*]"
echo "    dc     min/med/max : $(stats "${dcs[@]}")"
echo "    cuda   min/med/max : $(stats "${cus[@]}")"
awk -v a="$dmin" -v b="$cmin" 'BEGIN{printf "    ratio dc/cuda (min/min) : %.3f x\n", a/b}'

if [ "${#nas[@]}" -gt 0 ]; then
   nmin=$(printf '%s\n' "${nas[@]}" | sort -g | head -1)
   echo "    native min/med/max : $(stats "${nas[@]}")   <- cudaMalloc, no OpenACC"
   awk -v a="$cmin" -v b="$nmin" 'BEGIN{
      printf "    ratio cuda/native (min/min) : %.3f x   <- does host_data cost anything?\n", a/b}'
fi
