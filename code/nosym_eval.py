#!/usr/bin/env python3
"""Step 3: compare tensor scores from RAW DIRECTED NLI matrices (no symmetrisation) with the stored symmetrised ones on the same queries.
Inputs: <out-dir>/tensor_raw.jsonl (tensor_decompose.py on the --no-symmetrize S1/S2 files), ids.json.
Also checks that symmetrising the raw S1/S2 matrices reproduces the stored symmetrised matrices (max abs difference).
Usage (from /mnt/src): python nosym_eval.py --data-dir /mnt/output --cell coqa_llama3.1_8b --out-dir /mnt/output/nosym
"""
import argparse, json
from pathlib import Path
import numpy as np
from gate_eval import Cell
from baseline_auroc import official_auroc

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cell", required=True)
ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--n-boot", type=int, default=2000); a = ap.parse_args()
C = Cell(a.data_dir, a.cell, "cal"); ids = json.load(open(a.out_dir / "ids.json")); te = C.rows["test"]; pos = {str(i): k for k, i in enumerate(C.split["test"])}
raw = {}
for ln in open(a.out_dir / "tensor_raw.jsonl", encoding="utf-8"):
    if ln.strip():
        r = json.loads(ln); raw[str(r["id"])] = (r["U_CP"], r["U_Tucker"])
use = [i for i in ids if i in raw]; sel = np.array([pos[i] for i in use]); n = len(use)
cp_r, tk_r = np.array([raw[i][0] for i in use]), np.array([raw[i][1] for i in use]); cp_s, tk_s = C.rawCP[te][sel], C.rawTK[te][sel]
# matrix check
def load(p):
    d = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln); d[str(r["id"])] = r
    return d
for tag, raw_p, st_p in (("S1", a.out_dir / "s1_raw.jsonl", a.data_dir / f"stage_b_s1_semantic_{a.cell}.jsonl"), ("S2", a.out_dir / "s2_raw.jsonl", a.data_dir / f"stage_b_s2_claim_{a.cell}.jsonl")):
    R, S = load(raw_p), load(st_p); dm = 0.0; asym = 0.0
    for i in use:
        for key in ("matrix_agreement", "matrix_disagreement"):
            M = np.array(R[i][key]); asym = max(asym, float(np.abs(M - M.T).max()))
            Msym = np.array(R[i][key]); Msym = 0.5 * (Msym + Msym.T)
            dm = max(dm, float(np.abs(Msym - np.array(S[i][key])).max()))
    print(f"{tag}: max|sym(raw) - stored| = {dm:.2e}; max asymmetry of raw = {asym:.3f}")
rng = np.random.default_rng(1); idxs = [rng.integers(0, n, n) for _ in range(a.n_boot)]
print(f"n = {n}; Spearman(raw CP, sym CP) / Tucker:", end=" ")
from scipy.stats import spearmanr
print(round(spearmanr(cp_r, cp_s)[0], 4), round(spearmanr(tk_r, tk_s)[0], 4))
for thr in (90, 70):
    acc = C.acc[thr][te][sel]
    sets = {"CP": (cp_r, cp_s), "Tucker": (tk_r, tk_s), "Sum": (cp_r + tk_r, cp_s + tk_s)}
    for m, (r_, s_) in sets.items():
        ar, as_ = official_auroc(r_, acc, C.N), official_auroc(s_, acc, C.N)
        d = np.array([official_auroc(r_[i], acc[i], C.N) - official_auroc(s_[i], acc[i], C.N) for i in idxs]); lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"tau={thr} {m:6s} raw={ar:.4f} sym={as_:.4f} diff={ar-as_:+.4f} [{lo:+.4f}, {hi:+.4f}]")
