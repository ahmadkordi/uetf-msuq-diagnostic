#!/usr/bin/env bash
# Every analysis that exists for the Llama-2-13B BASE cells, run for the four Llama-2-13B CHAT cells.
# Run from /mnt/src with the venv active. Each step logs to /mnt/data/tmp/chat_full/<step>.log and the script goes on if one step fails.
# Base outputs are never overwritten: per-cell outputs have chat cell names; the few fixed-name outputs are moved aside and restored.
#   bash run_chat_full_analysis.sh check     # read-only checks
#   bash run_chat_full_analysis.sh <step>    # one of: cfg stats ceiling ceilingq expl extra kle klesp estb3 tsem tclean calib cmp
O=/mnt/output; E=$O/experiment_0b; G=$E/gate; L=/mnt/data/tmp/chat_full; mkdir -p $L
C="coqa_Llama2chat_13b hotpotqa_Llama2chat_13b nqopen_Llama2chat_13b truthfulqa_Llama2chat_13b"
MAXN=${MAXN:-0}          # tensor_sem_only --max-n; 0 = all test queries (set it to the value used for the base cells after 'check')
run() { local name=$1; shift; echo "=== $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee $L/$name.log | tail -${TAILN:-25}; }

case "$1" in
check)
  ls $G | grep -i "gate_config_coqa_Llama2_13b\|gate_chat\|tsem_coqa_Llama2_13b\|ceiling" | head -20
  head -3 $G/tsem_coqa_Llama2_13b.csv 2>/dev/null
  ls $E/stats_extra | head -12; ls $E/exploratory | head -4; ls $G | grep -c "extra_\|kle_\|estb3_\|tclean_\|tsem_" ;;
cfg)   # per-cell frozen configs for stats_extra, same format as the base ones ({cell: config}); from the chat validation run
  python - <<'EOF'
import json
d=json.load(open("/mnt/output/experiment_0b/gate/gate_chat_config_multi.json"))
for c,v in d.items():
    p=f"/mnt/output/experiment_0b/gate/gate_config_{c}.json"
    import os
    assert not os.path.exists(p), p
    json.dump({c:v},open(p,"w"),indent=2); print("wrote",p)
EOF
  ;;
stats)   # frozen-config re-evaluation with CIs (fixed output dir: base dir moved aside, chat result moved to stats_extra_chat, base restored)
  [ -d $E/stats_extra_base_tmp ] && { echo "stats_extra_base_tmp exists; stop"; exit 1; }
  mv $E/stats_extra $E/stats_extra_base_tmp
  python stats_extra.py --data-dir $O --cells $C 2>&1 | tee $L/stats_extra.log | tail -30
  mv $E/stats_extra $E/stats_extra_chat; mv $E/stats_extra_base_tmp $E/stats_extra
  ls $E/stats_extra | wc -l; ls $E/stats_extra_chat | head ;;
ceiling)  # supervised reference on VAL (diagnostic)
  run ceiling_u    python supervised_ceiling.py --data-dir $O --cells $C --features u    --tag ceiling_chat
  run ceiling_full python supervised_ceiling.py --data-dir $O --cells $C --features full --tag ceiling_chat ;;
ceilingq) run ceilingq python supervised_ceiling_q.py --data-dir $O --cells $C --n-boot 1000 ;;
expl)     run exploratory python exploratory_cv.py --data-dir $O --cells $C --folds 3 --n-boot 300 --seed 0 ;;
extra)    run extra python extra_baselines.py --data-dir $O --cells $C --n-boot 1000 --workers 8 ;;
kle)      run kle python kle_baseline.py --data-dir $O --cells $C --n-boot 1000 ;;
klesp)    # fixed output name kle_spearman.csv
  cp $G/kle_spearman.csv /mnt/data/tmp/kle_spearman_base.csv
  python kle_spearman.py --data-dir $O --cells $C 2>&1 | tee $L/kle_spearman.log | tail -20
  mv $G/kle_spearman.csv $G/kle_spearman_chat.csv; cp /mnt/data/tmp/kle_spearman_base.csv $G/kle_spearman.csv
  wc -l $G/kle_spearman.csv $G/kle_spearman_chat.csv ;;
estb3)    run estb3 python estimator_vs_b3.py --data-dir $O --cells $C --n-boot 1000 ;;
tsem)     run tsem python tensor_sem_only.py --data-dir $O --cells $C --max-n $MAXN --check 20 --n-boot 1000 --workers 16 ;;
tclean)   run tclean python tensor_clean_ablation.py --data-dir $O --cells $C --check 10 --n-boot 1000 --workers 16 ;;
calib)    python calibration_metrics.py --dir $E/stats_extra_chat 2>&1 | tee $L/calibration.log | tail -30 ;;   # writes calibration_metrics.csv INSIDE stats_extra_chat
cmp)      run base_vs_chat python compare_base_chat.py --base-dir $E/stats_extra --chat-dir $E/stats_extra_chat --out $E/base_vs_chat ;;
*) echo "unknown step"; exit 1 ;;
esac
