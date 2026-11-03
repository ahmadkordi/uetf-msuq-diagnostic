#!/usr/bin/env python3
"""Baseline reproduction: official-protocol AUROC of every raw estimator and of the MS-UQ
ensembles, at judge thresholds 90 (official code) and 70 (paper prose). EVALUATION-ONLY.

Protocol = the official MSUQ evaluation.py as read from the repo (not guessed):
  * per-query accuracy acc = mean binary correctness (float, not thresholded again)
  * labels are expanded to N per-response binary labels: true_number = int(acc*N) ones
    (int() TRUNCATES, as in the official code), the rest zeros
  * the query-level uncertainty U is copied to all N responses; NaN -> 0
  * ROC input is confidence = 1 - U       (official `seman_cca = 1 - y_predict`)
  implemented exactly via sample weights (t positives, N-t negatives per query).
Ensembles of the two tensor estimators: SUM = U_Tucker + U_CP (paper eq. 7, paper default),
MIN = min(U_Tucker, U_CP) (paper eq. 8; the shipped inter_unq_ensemble.py actually calls the
MIN variant -- a paper-vs-code discrepancy to document).
Uses RAW estimator values (stage_g_trainfit 'raw'): no fitting of any kind, so using ALL ids
is leakage-free for this baseline check; pass --ids to restrict.
AUARC is NOT computed: the official AUARC code looks inconsistent (reversed accuracy vs
unreversed x-axis); reproduce only after deciding how to treat that.

Paper Table 1 reference (threshold 70 in the paper's text; subset sizes unknown), AUROC:
  Llama2-13B  CoQA Sum .7802 P(true) .7102 | HotpotQA .6461 / .6056 | NQ .6435 / .6197
  Llama3.1-8B CoQA Sum .6111 P(true) .5816 | HotpotQA .6588 / .6362 | NQ .7765 / .7061
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

REF = {  # (cell prefix, model key) -> (MS-UQ-Sum, P(true))
    ("coqa", "Llama2_13b"): (.7802, .7102), ("hotpotqa", "Llama2_13b"): (.6461, .6056), ("nqopen", "Llama2_13b"): (.6435, .6197),
    ("coqa", "llama3.1_8b"): (.6111, .5816), ("hotpotqa", "llama3.1_8b"): (.6588, .6362), ("nqopen", "llama3.1_8b"): (.7765, .7061),
}


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln); out[str(r["id"])] = r
    return out


def official_auroc(u, acc, n):
    u = np.nan_to_num(np.asarray(u, float), nan=0.0)
    t = np.array([int(a * n) for a in acc], float)          # official int() truncation
    y = np.r_[np.ones(len(u)), np.zeros(len(u))]
    w = np.r_[t, n - t]
    s = np.r_[1 - u, 1 - u]
    keep = w > 0
    if len(np.unique(y[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y[keep], s[keep], sample_weight=w[keep]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output"))
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--ids", type=Path, nargs="*", default=None, help="optional id files to restrict to")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[90, 70])
    a = ap.parse_args()
    O = a.data_dir; D = O / "experiment_0b"
    cells = a.cells or sorted(p.name[len("stage_g_trainfit_"):-6] for p in D.glob("stage_g_trainfit_*.jsonl"))
    keep = None
    if a.ids:
        keep = set()
        for p in a.ids: keep |= {l.strip() for l in Path(p).read_text().splitlines() if l.strip()}
    rows, lines = [], []
    for cell in cells:
        cal = load(D / f"stage_g_trainfit_{cell}.jsonl"); corr = load(O / f"stage_h_correctness_{cell}.jsonl")
        ids = sorted(set(cal) & set(corr) & (keep if keep is not None else set(cal)))
        names = sorted(cal[ids[0]]["raw"])
        U = {n: np.array([cal[i]["raw"][n] for i in ids], float) for n in names}
        cp, tk = "tensor.U_CP", "tensor.U_Tucker"
        if cp in U and tk in U:
            U["ENS.SUM(Tucker+CP)"] = U[tk] + U[cp]; U["ENS.MIN(Tucker,CP)"] = np.minimum(U[tk], U[cp])
        lines.append(f"\n== {cell}  (n_queries={len(ids)}) ==")
        acc_by_thr = {}
        for thr in a.thresholds:
            acc, N = [], None
            for i in ids:
                pr = corr[i]["per_response"]; N = len(pr)
                c = [1 if r["score"] > thr else 0 for r in pr]
                acc.append(sum(c) / len(c))
            acc = np.array(acc); acc_by_thr[thr] = (acc, N)
            lines.append(f"  threshold {thr:g}: mean acc={acc.mean():.3f}  frac queries acc==0: {(acc==0).mean():.3f}  acc==1: {(acc==1).mean():.3f}")
        hdr = "  " + "estimator".ljust(30) + "".join(f"AUROC@{t:g}".rjust(12) for t in a.thresholds)
        lines.append(hdr)
        for n, u in U.items():
            vals = [official_auroc(u, *acc_by_thr[t]) for t in a.thresholds]
            lines.append("  " + n.ljust(30) + "".join(f"{v:12.4f}" for v in vals))
            rows.append([cell, n, len(ids)] + [f"{v:.4f}" for v in vals])
        pref, _, model = cell.partition("_")
        if (pref, model) in REF:
            s, p = REF[(pref, model)]
            lines.append(f"  [paper Table 1] MS-UQ-Sum={s:.4f}  P(true)={p:.4f}   (compare ENS.SUM and s3_self_eval.U_p; paper threshold 70)")
    txt = "\n".join(lines); print(txt)
    (D / "baseline_auroc.txt").write_text(txt)
    with open(D / "baseline_auroc.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["cell", "estimator", "n"] + [f"auroc@{t:g}" for t in a.thresholds]] + rows)


if __name__ == "__main__":
    main()
