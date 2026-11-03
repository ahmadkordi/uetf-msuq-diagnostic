#!/usr/bin/env bash
# Collects the remaining small result files into one text file (read-only).
cd /mnt/output/experiment_0b || exit 1
OUT=/mnt/data/tmp/bundle3.txt
: > "$OUT"
for f in gate/ceiling_chat_u_multi.txt gate/ceiling_chat_full_multi.txt gate/ceiling_u_coqa_Llama2_13b.txt gate/ceiling_u_hotpotqa_Llama2_13b.txt gate/ceiling_u_nqopen_Llama2_13b.txt gate/ceiling_u_truthfulqa_Llama2_13b.txt gate/gate_chat_val_multi.txt gate/gate_chat_test_multi.txt gate/kle_spearman.csv stats_extra_chat/auroc_ci_by_cell.csv stats_extra_chat/contrasts_by_cell.csv stats_extra/auroc_ci_by_cell.csv stats_extra/contrasts_by_cell.csv ../experiment_0a/dependency_audit_summary.md; do
  echo "######## $f" >> "$OUT"
  if [ -f "$f" ]; then cat "$f" >> "$OUT"; else echo "MISSING: $f" >> "$OUT"; fi
done
wc -c "$OUT"; grep -c "^MISSING" "$OUT"; grep "^MISSING" "$OUT"
