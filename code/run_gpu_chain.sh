#!/bin/bash
# usage: bash run_gpu_chain.sh <dataset> <gpu>   (cell = <dataset>_Llama2chat_13b)
ds=$1; g=$2; c=${ds}_Llama2chat_13b; O=/mnt/output; M=/mnt/data/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source ~/venvs/uncertainty/bin/activate; cd /mnt/src
a=$O/stage_a_$c.jsonl; n=$(grep -c . $a); cl=$O/stage_b_claims_$c.jsonl
sk(){ [ -s "$1" ]; }
sk $O/stage_b_s1_semantic_$c.jsonl || python semantic_nli.py --input $a --output $O/stage_b_s1_semantic_$c.jsonl --model $M/deberta-large-mnli --gpu-id $g || exit 1
sk $O/stage_b_s3_self_eval_$c.jsonl || python self_eval.py --input $a --output $O/stage_b_s3_self_eval_$c.jsonl --model $M/Llama-2-13b-chat-hf --gpu-id $g || exit 1
while [ "$(grep -c . $cl 2>/dev/null)" != "$n" ]; do echo "$(date +%T) waiting for claims ($(grep -c . $cl 2>/dev/null)/$n)"; sleep 300; done
sk $O/stage_b_s2_claim_$c.jsonl || python claim_nli.py --input $cl --output $O/stage_b_s2_claim_$c.jsonl --model $M/deberta-large-mnli --gpu-id $g || exit 1
sk $O/stage_c_tensor_$c.jsonl || python tensor_decompose.py --s1 $O/stage_b_s1_semantic_$c.jsonl --s2 $O/stage_b_s2_claim_$c.jsonl --output $O/stage_c_tensor_$c.jsonl || exit 1
sk $O/stage_f_estimators_$c.jsonl || python graph_estimators.py --s1 $O/stage_b_s1_semantic_$c.jsonl --s2 $O/stage_b_s2_claim_$c.jsonl --s3 $O/stage_b_s3_self_eval_$c.jsonl --output $O/stage_f_estimators_$c.jsonl || exit 1
echo "$ds chain DONE"
