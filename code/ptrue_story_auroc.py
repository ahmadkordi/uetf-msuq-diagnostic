#!/usr/bin/env python3
"""Does adding the story to the self-evaluation prompt change the P(true) AUROC? (own test, same production batched function)
For a fixed random subset of the TEST split of one CoQA cell: U_p = 1 - mean_i P_true(i) over the N=20 responses, computed
  (a) as in production (story=""), (b) with story = record['story_so_far'];
official AUROC (int(acc*N) truncation) at tau 90 and 70, paired query-bootstrap of (b) - (a).
Usage (from /mnt/src): HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ptrue_story_auroc.py <cell> <model_path> [max_n=600]
Example: python ptrue_story_auroc.py coqa_Llama2_13b /mnt/data/models/Llama-2-13b-hf 600
"""
import sys, json
from pathlib import Path
import numpy as np
import self_eval as se
from gate_eval import Cell
from baseline_auroc import official_auroc

cell, model_path = sys.argv[1], sys.argv[2]; max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 600
O = Path("/mnt/output"); C = Cell(O, cell, "cal"); te = C.rows["test"]; ids = list(C.split["test"])
rec = {}
for l in open(O / f"stage_a_{cell}.jsonl"):
    if l.strip():
        r = json.loads(l); rec[str(r["id"])] = r
sel = np.arange(len(ids))
if max_n < len(ids): sel = np.sort(np.random.default_rng(7).choice(len(ids), max_n, replace=False))
tok, model, dev, a, b = se.load_eval_model(model_path)
txt = lambda x: x if isinstance(x, str) else x["text"]
U0, U1 = [], []
for k, i in enumerate(sel):
    r = rec[str(ids[i])]; texts = [txt(x) for x in r["responses"]]
    p0 = se.p_true_for_responses_batch(r["question"], texts, tokenizer=tok, model=model, device=dev, a_tok=a, b_tok=b, story="", batch_size=20)
    p1 = se.p_true_for_responses_batch(r["question"], texts, tokenizer=tok, model=model, device=dev, a_tok=a, b_tok=b, story=r.get("story_so_far") or "", batch_size=20)
    U0.append(1 - float(np.mean(p0))); U1.append(1 - float(np.mean(p1)))
    if k % 100 == 0: print(k, flush=True)
U0, U1 = np.array(U0), np.array(U1); n = len(sel)
rng = np.random.default_rng(1); idxs = [rng.integers(0, n, n) for _ in range(1000)]
for thr in (90, 70):
    acc = C.acc[thr][te][sel]
    a0, a1 = official_auroc(U0, acc, C.N), official_auroc(U1, acc, C.N)
    d = np.array([official_auroc(U1[i], acc[i], C.N) - official_auroc(U0[i], acc[i], C.N) for i in idxs])
    print(f"{cell} tau={thr} n={n}: AUROC no-story {a0:.4f} | with story {a1:.4f} | diff {a1-a0:+.4f} [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")
print("query-level U_p sd: no-story %.4f, with-story %.4f" % (U0.std(), U1.std()))
