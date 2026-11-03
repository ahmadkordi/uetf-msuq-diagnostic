#!/usr/bin/env python3
"""P(true) with vs without the story/context in the self-evaluation prompt (CoQA records have 'story_so_far').
Same code path as production (self_eval._build_prompt, same A/B tokens, bf16, single-prompt forward passes), only `story` differs.
Usage (from /mnt/src): HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ptrue_story_test.py /mnt/output/stage_a_coqa_Llama2_13b.jsonl /mnt/data/models/Llama-2-13b-hf [n_queries=20]
"""
import sys, json, torch, numpy as np
from scipy.stats import spearmanr
import self_eval as se

path, model_path = sys.argv[1], sys.argv[2]
nq = int(sys.argv[3]) if len(sys.argv) > 3 else 20
rows = [json.loads(l) for l in open(path)][:nq]
tok, model, dev, a, b = se.load_eval_model(model_path)
txt = lambda x: x if isinstance(x, str) else x["text"]

def p_true(prompt):
    enc = tok(prompt, return_tensors="pt").to(dev)
    with torch.inference_mode():
        lg = model(**enc).logits[0, -1, :].float()
    return float(torch.softmax(lg[[a, b]], -1)[0])

no_story, with_story, slen = [], [], []
for r in rows:
    ans = [txt(x) for x in r["responses"]][:5]; story = r.get("story_so_far") or ""
    slen.append(len(story))
    for i, x in enumerate(ans):
        oth = ans[:i] + ans[i + 1:]
        no_story.append(p_true(se._build_prompt(r["question"], x, oth, story="", max_brainstorm=10)))
        with_story.append(p_true(se._build_prompt(r["question"], x, oth, story=story, max_brainstorm=10)))
n0, n1 = np.array(no_story), np.array(with_story)
print("story chars: min %d median %d max %d" % (min(slen), np.median(slen), max(slen)))
print("no story  : mean %.3f sd %.3f distinct %d" % (n0.mean(), n0.std(), len(set(np.round(n0, 6)))))
print("with story: mean %.3f sd %.3f distinct %d" % (n1.mean(), n1.std(), len(set(np.round(n1, 6)))))
print("Spearman no-story vs story: %.3f | mean |diff| %.3f" % (spearmanr(n0, n1)[0], abs(n0 - n1).mean()))
