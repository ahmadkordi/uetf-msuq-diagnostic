#!/usr/bin/env python3
"""Cheap label-free baselines that need no new model calls: LexSim, NumSets and a discrete Semantic-Entropy approximation.

NO FITTING. Evaluated on the TEST split with the official-protocol AUROC, like B2/B3 (nothing is selected on test).
  LexSim     U = 1 - mean pairwise ROUGE-L F1 over the N=20 responses (lower-cased, whitespace tokens, own LCS implementation).
  NumSets    U = number of connected components of the graph with edge (i,j) iff matrix_agreement[i,j] > TH (default 0.5)
             (own approximation of NLI-based semantic sets using the stored, symmetrised agreement matrix; TH fixed a priori,
              sensitivity at 0.3 and 0.7 is also reported, never used to pick a winner).
  SE-disc    U = entropy of the component-size distribution (equal sample weights) = discrete semantic entropy on the same clusters.
Deviations to state in the paper: ROUGE-L tokenisation is our own; clustering uses the symmetrised agreement matrix, not bidirectional
entailment of the original NumSets/SE; the agreement matrix definition is the one used in our pipeline.
Also computes B1 (best single estimator on TRAIN at the same threshold), B2 and B3-Sum for paired contrasts.
Usage (from /mnt/src):  python extra_baselines.py --data-dir /mnt/output --cells <cell> ... --n-boot 1000 --workers 8
Output: <data-dir>/experiment_0b/gate/extra_<cell>.csv
"""
from __future__ import annotations
import argparse, csv, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from gate_eval import Cell
from baseline_auroc import official_auroc

try:
    from rapidfuzz.distance import LCSseq
    def lcs(a, b): return LCSseq.similarity(a, b)
except ImportError:  # slow fallback
    def lcs(a, b):
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b): cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        return prev[-1]


def rouge_l(a, b):
    if not a or not b: return 0.0
    l = lcs(a, b)
    if l == 0: return 0.0
    p, r = l / len(b), l / len(a); return 2 * p * r / (p + r)


TEXT_KEYS = ("text", "response", "answer", "output", "content", "generated_text", "completion")


def as_text(r):
    if isinstance(r, str): return r
    if isinstance(r, dict):
        for k in TEXT_KEYS:
            if isinstance(r.get(k), str): return r[k]
        raise KeyError(f"no text key in response dict; keys = {list(r)}")
    return str(r)


def lexsim(resps):
    T = [as_text(r).lower().split() for r in resps]; n = len(T); s = [rouge_l(T[i], T[j]) for i in range(n) for j in range(i + 1, n)]
    return 1.0 - float(np.mean(s))


def components(A, th):
    n = len(A); lab = -np.ones(n, int); c = 0
    for i in range(n):
        if lab[i] >= 0: continue
        st = [i]; lab[i] = c
        while st:
            u = st.pop()
            for v in np.where((A[u] > th) | (A[:, u] > th))[0]:
                if lab[v] < 0: lab[v] = c; st.append(v)
        c += 1
    return lab


def one(args):
    resps, A = args; A = np.array(A, float); out = {"LexSim": lexsim(resps)}
    for th in (0.5, 0.3, 0.7):
        lab = components(A, th); sizes = np.bincount(lab) / len(lab)
        tag = "" if th == 0.5 else f"@{th}"
        out[f"NumSets{tag}"] = float(len(sizes)); out[f"SE-disc{tag}"] = float(-(sizes * np.log(sizes)).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    O = a.data_dir; out = O / "experiment_0b" / "gate"
    for cell in a.cells:
        C = Cell(O, cell, "cal"); te, tr = C.rows["test"], C.rows["train"]; ids = C.split["test"]; recs = {}
        with open(O / f"stage_b_s1_semantic_{cell}.jsonl", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    recs[str(r["id"])] = (r["responses"], r["matrix_agreement"])
        with Pool(a.workers) as p: res = p.map(one, [recs[i] for i in ids], chunksize=32)
        U = {k: np.array([r[k] for r in res]) for k in res[0]}
        b1 = {thr: max(range(C.K), key=lambda k: official_auroc(C.U[tr][:, k], C.acc[thr][tr], C.N)) for thr in (90, 70)}
        ref = {"B2": C.U[te].mean(1), "B3-Sum": (C.rawTK + C.rawCP)[te]}
        rng = np.random.default_rng(1); n = len(te); idxs = [rng.integers(0, n, n) for _ in range(a.n_boot)]; rows = []
        for thr in (90, 70):
            acc = C.acc[thr][te]; ref_t = dict(ref); ref_t["B1"] = C.U[te][:, b1[thr]]
            allm = {**U, **ref_t}
            for m, u in U.items():
                row = {"cell": cell, "thr": thr, "method": m, "auroc": official_auroc(u, acc, C.N)}
                bs = np.array([official_auroc(u[i], acc[i], C.N) for i in idxs]); row["lo"], row["hi"] = np.percentile(bs, [2.5, 97.5])
                for rname, ru in ref_t.items():
                    d = row["auroc"] - official_auroc(ru, acc, C.N)
                    bd = bs - np.array([official_auroc(ru[i], acc[i], C.N) for i in idxs])
                    row[f"d_{rname}"] = d; row[f"lo_{rname}"], row[f"hi_{rname}"] = np.percentile(bd, [2.5, 97.5])
                rows.append(row)
            for rname, ru in ref_t.items():  # also store the reference AUROCs for the table
                rows.append({"cell": cell, "thr": thr, "method": rname, "auroc": official_auroc(ru, acc, C.N)})
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("cell", "thr", "method", "auroc"), k))
        with open(out / f"extra_{cell}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"[{cell}] done", flush=True)


if __name__ == "__main__":
    main()
