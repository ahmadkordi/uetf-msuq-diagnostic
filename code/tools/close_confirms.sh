#!/usr/bin/env bash
# Read-only checks that close the [CONFIRM] items. Run from /mnt/src with the venv active:  bash close_confirms.sh
# Everything goes to /mnt/data/tmp/confirm_check.txt (send that file back).
OUT=/mnt/data/tmp/confirm_check.txt; O=/mnt/output; S=/mnt/src
{
echo "##### 1. tokenizer call in the generation scripts (CoQA, HotpotQA, TruthfulQA, NQ-Open) and in self_eval"
grep -n "AutoTokenizer.from_pretrained\|LlamaTokenizer" $S/generate_responses_coqa_shard.py $S/generate_responses_hotpotqa_v3_share_task.py $S/generate_responses_truthfulqa_shard.py $S/generate_responses_nqopen_shard.py $S/self_eval.py
echo; echo "##### 2. generation settings stored in stage_a records (all cells): distinct settings and number of records"
python3 - <<'PY'
import json, glob, collections
for f in sorted(glob.glob("/mnt/output/stage_a_*.jsonl")):
    c = collections.Counter(); n = 0
    for l in open(f):
        if not l.strip(): continue
        n += 1
        g = json.loads(l).get("generation")
        if isinstance(g, dict): g = {k: v for k, v in g.items() if k != "gpu_id"}
        c[json.dumps(g, sort_keys=True)] += 1
    print(f.split("/")[-1], "records", n, "distinct settings", len(c))
    for k, v in c.most_common(3): print("   ", v, k)
PY
echo; echo "##### 3. which script produced the stage-B S3 (self-evaluation) files"
ls -l --time-style=full-iso $S/self_eval*.py
find $O -maxdepth 2 \( -iname "*s3*" -o -iname "*self_eval*" -o -iname "*ptrue*" -o -iname "*p_true*" \) | head -40
grep -rl "self_eval" /mnt/data/tmp --include=*.log --include=*.txt 2>/dev/null | head -20
echo; echo "##### 4. precision of the self-evaluation forward pass"
grep -n "float32\|float16\|bfloat16\|torch_dtype\|dtype=" $S/self_eval.py $S/self_eval_shard.py $S/self_eval_share.py 2>/dev/null | head -30
grep -rn "fp32\|float32" /mnt/data/tmp --include=*.log 2>/dev/null | head -10
echo; echo "##### 5. models and quantisation recorded for every cell (see also section 2)"
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("/mnt/output/stage_a_*.jsonl")):
    g = json.loads(open(f).readline()).get("generation"); print(f.split("/")[-1], {k: g.get(k) for k in ("model", "quantize_4bit", "dtype")} if isinstance(g, dict) else g)
PY
echo; echo "##### 6. claim-extraction model and judge model defaults"
grep -n "qwen" $S/extract_claims.py $S/score_correctness.py | head -12
echo; echo "##### 7. judge call: model name and Ollama options"
grep -n "temperature\|num_predict\|seed\|model" $S/score_correctness.py | head -20
} > $OUT 2>&1
wc -l $OUT; echo "send $OUT"
