#!/usr/bin/env bash
# Third read-only round (precision of the self-evaluation, Ollama quantisation).  bash close_confirms3.sh -> /mnt/data/tmp/confirm_check3.txt
OUT=/mnt/data/tmp/confirm_check3.txt; S=/mnt/src
{
echo "##### a. p_true distribution table of the diagnostic (rows)"
sed -n 38,80p /mnt/data/tmp/diag_self_eval.txt
echo; echo "##### b. dtype and softmax precision in self_eval.py (defaults and conversions)"
grep -n "dtype\|\.float()\|softmax\|add_argument(\"--dtype\|add_argument(\"--quantize" $S/self_eval.py | cut -c1-200
grep -n "add_argument(\"--dtype\|add_argument('--dtype\|\.float()\|softmax" $S/self_eval_shard.py $S/self_eval_share.py | cut -c1-200
echo; echo "##### c. Ollama builds"
ollama show qwen2.5:32b 2>&1 | head -20
ollama show qwen:32b 2>&1 | head -20
} > $OUT 2>&1
wc -l $OUT
