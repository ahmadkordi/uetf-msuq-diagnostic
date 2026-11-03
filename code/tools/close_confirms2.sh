#!/usr/bin/env bash
# Second read-only round for the remaining [CONFIRM] items.  Run from /mnt/src:  bash close_confirms2.sh   -> /mnt/data/tmp/confirm_check2.txt
OUT=/mnt/data/tmp/confirm_check2.txt; O=/mnt/output; S=/mnt/src
{
echo "##### A. every generation script and its tokenizer call (nq-open launcher vs worker)"
ls -l --time-style=full-iso $S/generate_responses*.py
grep -n "AutoTokenizer\|tokenizer = \|LlamaTokenizer" $S/generate_responses*.py | head -40
echo; echo "##### B. S3 files: modification time and record keys"
python3 - <<'PY'
import glob, json, os, time
for f in sorted(glob.glob("/mnt/output/stage_b_s3*.jsonl")):
    l = open(f).readline()
    try: r = json.loads(l)
    except Exception: r = {}
    keys = list(r.keys()) if isinstance(r, dict) else []
    extra = {k: r[k] for k in keys if any(w in k.lower() for w in ("dtype", "script", "model", "version", "fp", "precision", "quant"))}
    print(os.path.basename(f), time.strftime("%Y-%m-%d %H:%M", time.gmtime(os.path.getmtime(f))), "keys:", keys[:14], extra)
PY
echo; echo "##### C. shell history lines that ran the self-evaluation"
grep -h "self_eval" ~/.bash_history 2>/dev/null | head -40
echo; echo "##### D. the self-evaluation diagnostic file (first 40 lines)"
head -40 /mnt/data/tmp/diag_self_eval.txt 2>/dev/null
echo; echo "##### E. duplicate chat files with old names (same content?)"
ls -l --time-style=full-iso $O/stage_a_coqa_llama2_13b_chat.jsonl $O/stage_a_coqa_Llama2chat_13b.jsonl $O/stage_a_nqopen_llama2_13b_chat.jsonl $O/stage_a_nqopen_Llama2chat_13b.jsonl $O/stage_a_truthfulqa_Llama2_13b_chat.jsonl $O/stage_a_truthfulqa_Llama2chat_13b.jsonl 2>&1
cmp $O/stage_a_coqa_llama2_13b_chat.jsonl $O/stage_a_coqa_Llama2chat_13b.jsonl && echo "coqa duplicate identical"
cmp $O/stage_a_nqopen_llama2_13b_chat.jsonl $O/stage_a_nqopen_Llama2chat_13b.jsonl && echo "nqopen duplicate identical"
cmp $O/stage_a_truthfulqa_Llama2_13b_chat.jsonl $O/stage_a_truthfulqa_Llama2chat_13b.jsonl && echo "truthfulqa duplicate identical"
} > $OUT 2>&1
wc -l $OUT
