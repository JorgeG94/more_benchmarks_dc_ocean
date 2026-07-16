#!/bin/bash
# Sweep the real configs, min-of-N per variant. MIN, not mean: the shared HPC
# analysis node is shared and a co-tenant job produces occasional large
# outliers on whichever variant happens to run when it lands.
#
# nx/ny are POINT COUNTS from ~/analysis_gebco/*.nml; the driver adds nghost=3.
N=${N:-2}
printf "%-22s %9s %9s %9s %9s %9s %9s %9s\n" \
  "config (cells)" "dc" "dc_async" "dc_fix" "fused" "fus+asy" "cu_faith" "cu_fused"
for cfg in "108 137 30" "473 297 30" "240 560 20" "945 594 30" \
           "473 297 15" "473 297 50" "473 297 75"; do
  best=""; cells=""
  for r in $(seq 1 $N); do
    o=$(./ale_bench $cfg 12 8 2>&1)
    [ -z "$cells" ] && cells=$(echo "$o" | grep -oP '=> \K[0-9]+(?= cells)')
    v=$(echo "$o" | grep -E "^  (DC |CUDA )" | grep -oP '^\s+\S.*?\s+\K[0-9]+\.[0-9]+' | tr '\n' ' ')
    if [ -z "$best" ]; then best="$v"; else
      best=$(paste -d' ' <(echo $best|tr ' ' '\n') <(echo $v|tr ' ' '\n') \
             | awk 'NF==2{print ($2<$1)?$2:$1}' | tr '\n' ' ')
    fi
  done
  printf "%-22s %s\n" "$(echo $cfg|tr ' ' 'x') ($cells)" \
    "$(echo $best|awk '{printf "%9s %9s %9s %9s %9s %9s %9s",$1,$2,$3,$4,$5,$6,$7}')"
done
