#!/bin/bash
# Paired timing: Fortran-driven CUDA (host_data) vs native CUDA (cudaMalloc).
#
# Answers one question -- does the OpenACC runtime on the CUDA path cost
# anything? -- on a V100 that up to 9 sibling agents may be sharing.
#
# CLEAN-WINDOW SAMPLING, because contention here is intermittent (idle for a few
# seconds, then 5 siblings) and a contended number is worthless: a DC number
# read 14.88 ms contended vs 3.52 ms idle. Naive approaches that FAIL:
#   * one run each             -> whoever ran in the idle window wins.
#   * min-across-repeats, but  -> the binary that runs FIRST in each repeat
#     always the same order       systematically grabs the idle window. The
#                                 first version of this script did exactly that
#                                 and made native look 1.5x slower purely from
#                                 running second. Alternating the order is not
#                                 optional.
# So: each sample takes ONE binary, checks the GPU is idle immediately BEFORE
# and again AFTER, and is DISCARDED unless both checks pass. Only samples that
# had the GPU to themselves for their whole run are kept. Order alternates.
# Short runs (default 50 reps) so a sample fits inside a typical idle window.
set -u
ARGS="${ARGS:-473 297 24 50 10}"
NSAMP="${NSAMP:-24}"
cd "$(dirname "$0")"

busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
wait_idle() { for _ in $(seq 1 120); do busy || return 0; sleep 2; done; return 1; }

rm -f clean_f.txt clean_n.txt
nf=0; nn=0
for s in $(seq 1 "$NSAMP"); do
    # alternate which binary goes first, so neither is systematically favoured
    if [ $((s % 2)) -eq 0 ]; then who=f; else who=n; fi

    wait_idle || { echo "sample $s: no idle window, skipping"; continue; }
    busy && continue
    if [ "$who" = f ]; then
        out=$(./btstep_bench $ARGS 2>/dev/null | grep -E '^  CUDA')
    else
        out=$(./btstep_native $ARGS 2>/dev/null | grep -E '^  CUDA')
    fi
    if busy; then
        echo "sample $s ($who): GPU went busy mid-run -- DISCARDED"
        continue
    fi
    if [ "$who" = f ]; then echo "$out" >> clean_f.txt; nf=$((nf+1));
    else                    echo "$out" >> clean_n.txt; nn=$((nn+1)); fi
    echo "sample $s ($who): clean"
done

echo
echo "=== CLEAN SAMPLES: fortran=$nf  native=$nn   (args: $ARGS) ==="
pick() {  # $1=file $2=label-regex -> min of the first float on matching lines
    awk -v pat="$2" '$0 ~ pat {
        for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+$/) { v=$i+0; break }
        if (m=="" || v<m) m=v
    } END { if (m=="") print "n/a"; else printf "%.3f", m }' "$1" 2>/dev/null
}
printf "%-16s %12s %12s %10s\n" "mode" "fortran" "native" "ratio"
for m in faithful fused graph; do
    f=$(pick clean_f.txt "CUDA $m"); n=$(pick clean_n.txt "CUDA $m")
    if [ "$f" = n/a ] || [ "$n" = n/a ]; then
        printf "%-16s %12s %12s %10s\n" "$m" "$f" "$n" "n/a"
    else
        printf "%-16s %12s %12s %10s\n" "$m" "$f" "$n" \
               "$(awk -v a="$f" -v b="$n" 'BEGIN{printf "%.3f x", a/b}')"
    fi
done
echo
echo "min-of-clean-samples, ms/call, each a mean over the run's reps."
echo "ratio > 1 => native faster => the OpenACC path was costing something."
