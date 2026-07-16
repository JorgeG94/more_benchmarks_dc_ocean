#!/bin/bash
# Sweep the real configs, min-of-N per variant. MIN, not mean: the shared HPC
# analysis node is shared and a co-tenant job produces occasional 20-40x
# outliers on whichever variant runs when it lands (seen: a 7.5 ms `dc` at a
# config that reproducibly measures 0.187). Min-of-N is the robust statistic
# for "how fast is this kernel", which is the question.
N=${N:-3}
printf "%-24s %8s %8s %8s %8s %8s %8s %8s %8s\n" \
  "config (cells)" "dc" "dc_DT" "dc_fus" "dc_acc" "dc_f+a" "cu_16" "cu_6" "cu_gr"
for cfg in "108 137 30" "473 297 30" "240 560 30" "359 458 30" "945 594 30" \
           "473 297 1" "473 297 75" "1890 1188 30"; do
  best=""; cells=""
  for r in $(seq 1 $N); do
    o=$(./meke_bench $cfg 200 20 2>&1)
    [ -z "$cells" ] && cells=$(echo "$o" | grep -oP '= \K[0-9]+(?= cells)')
    v=$(echo "$o" | grep -E "^  DC   \(|DC \+ derived|DC \+ FUSED \(|DC \+ ACC|DC \+ FUSED \+ ACC|CUDA faithful|CUDA fused|CUDA graph" \
        | grep -oP ':\s+\K[0-9.]+' | tr '\n' ' ')
    if [ -z "$best" ]; then best="$v"; else
      best=$(paste -d' ' <(echo $best|tr ' ' '\n') <(echo $v|tr ' ' '\n') \
             | awk '{print ($2<$1)?$2:$1}' | tr '\n' ' ')
    fi
  done
  printf "%-24s %s\n" "$(echo $cfg|tr ' ' 'x') ($cells)" \
    "$(echo $best|awk '{printf "%8s %8s %8s %8s %8s %8s %8s %8s",$1,$2,$3,$4,$5,$6,$7,$8}')"
done
