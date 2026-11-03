#!/usr/bin/env bash
# Cause tests for base-vs-chat differences. Run from /mnt/src with the venv active.  Usage: bash run_chat_cause_tests.sh cpu|gpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
O=/mnt/output/experiment_0b/base_vs_chat; L=/mnt/data/tmp/chat_causes; mkdir -p $O $L
CH=/mnt/data/models/Llama-2-13b-chat-hf; BA=/mnt/data/models/Llama-2-13b-hf
case "$1" in
cpu)  python base_chat_causes.py --out $O 2>&1 | tee $L/causes_cpu.log | tail -80
      python token_test.py $BA $CH 2>&1 | tee $L/token_test.log | tail -40 ;;
gpu)  # P(true) with/without story on the CHAT cells (same function as production; 600 random test queries each)
      for c in coqa_Llama2chat_13b; do python ptrue_story_auroc.py $c $CH 600 2>&1 | tee $L/ptrue_story_$c.log | tail -6; done
      python ptrue_story_test.py /mnt/output/stage_a_coqa_Llama2chat_13b.jsonl $CH 20 2>&1 | tee $L/ptrue_story_test_chat.log | tail -15 ;;
*) echo "cpu|gpu"; exit 1 ;;
esac
