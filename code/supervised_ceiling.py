#!/usr/bin/env python3
"""Supervised CEILING diagnostic (NOT a v9 method): how much can ANY fusion of the estimator pool gain?

Fits a label-using model on TRAIN queries only (correctness at threshold T), evaluates on VAL with the
official-protocol AUROC. Purpose: bound the headroom of the pool. If even supervised fusion cannot beat
the best single estimator (B1) or MS-UQ (B3), no label-free gate can; if it can, the headroom exists and
the self-supervised target is what fails to find it. Never use as a proposed method; report as an
'oracle-supervised reference' only.
Models: logistic regression (C chosen on an internal 80/20 split of TRAIN) and histogram gradient boosting.
Inputs: tilde-U vector (+ with --features full: mu/sigma/range/DC_i and phi_Q-lite). Per-response labels are
reproduced exactly as in the official protocol via sample weights (t positives, N-t negatives per query).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from gate_eval import Cell, paired_boot
from baseline_auroc import official_auroc


def expand(X, acc, N):
    t = np.array([int(a * N) for a in acc], float)             # official int() truncation; t correct responses
    Xe = np.vstack([X, X]); y = np.r_[np.zeros(len(X)), np.ones(len(X))]   # target = INCORRECT (uncertainty)
    w = np.r_[t, N - t] * 0 + np.r_[t, N - t]
    # row block 1 (y=0, correct) weight t ; block 2 (y=1, incorrect) weight N-t
    w = np.r_[t, N - t]; k = w > 0
    return Xe[k], y[k], w[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--variant", default="cal"); ap.add_argument("--features", default="u", choices=["u", "full"])
    ap.add_argument("--n-boot", type=int, default=500); ap.add_argument("--tag", default="ceiling")
    a = ap.parse_args(); O = a.data_dir; out = O / "experiment_0b" / "gate"; out.mkdir(parents=True, exist_ok=True); lines = []
    for cell in a.cells:
        C = Cell(O, cell, a.variant); tr, va = C.rows["train"], C.rows["val"]
        X = C.U if a.features == "u" else np.hstack([C.U, C.U.mean(1, keepdims=True), C.U.std(1, keepdims=True), C.U.max(1, keepdims=True) - C.U.min(1, keepdims=True),
                                                    np.abs(C.U[:, :, None] - C.U[:, None, :]).sum(2) / (C.K - 1), C.Q])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - mu) / sd
        lines.append(f"\n== {cell} (VAL, supervised ceiling, features={a.features}) nTrain={len(tr)} nVal={len(va)} ==")
        lines.append(f"  {'method':14s} {'AUROC@90':>9s} {'AUROC@70':>9s}")
        res = {}
        for thr in (90, 70):
            acc = C.acc[thr]; rng = np.random.default_rng(0); perm = rng.permutation(len(tr)); cut = int(0.8 * len(tr)); a_idx, b_idx = tr[perm[:cut]], tr[perm[cut:]]
            best = (-1, 1.0)
            for Cc in (0.01, 0.1, 1.0, 10.0):
                Xe, y, w = expand(Xs[a_idx], acc[a_idx], C.N); m = LogisticRegression(C=Cc, max_iter=500).fit(Xe, y, sample_weight=w)
                s = C.auroc(m.predict_proba(Xs[b_idx])[:, 1], b_idx, thr)
                if s > best[0]: best = (s, Cc)
            Xe, y, w = expand(Xs[tr], acc[tr], C.N); lr = LogisticRegression(C=best[1], max_iter=500).fit(Xe, y, sample_weight=w)
            gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, early_stopping=True, random_state=0).fit(Xe, y, sample_weight=w)
            b1 = max(range(C.K), key=lambda k: official_auroc(C.U[tr][:, k], acc[tr], C.N))
            res[thr] = {"LR": lr.predict_proba(Xs[va])[:, 1], "GB": gb.predict_proba(Xs[va])[:, 1], "B1": C.U[va][:, b1], "B2": C.U[va].mean(1), "B3-Sum": (C.rawTK + C.rawCP)[va], "_b1": C.names[b1], "_C": best[1]}
        for m in ("B1", "B2", "B3-Sum", "LR", "GB"):
            note = res[90]["_b1"] + "/" + res[70]["_b1"] if m == "B1" else (f"C90={res[90]['_C']} C70={res[70]['_C']}" if m == "LR" else "")
            lines.append(f"  {m:14s} {C.auroc(res[90][m], va, 90):9.4f} {C.auroc(res[70][m], va, 70):9.4f}  {note}")
        for A, B in (("LR", "B1"), ("GB", "B1"), ("LR", "B3-Sum"), ("GB", "B3-Sum"), ("GB", "LR")):
            for thr in (90, 70):
                d = C.auroc(res[thr][A], va, thr) - C.auroc(res[thr][B], va, thr); lo, hi = paired_boot(C, va, thr, res[thr][A], res[thr][B], a.n_boot, 1)
                lines.append(f"  paired {A} - {B} @ {thr}: {d:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
        print("\n".join(lines[-18:]), flush=True)
    (out / f"{a.tag}_{a.features}_{'_'.join(a.cells) if len(a.cells) < 3 else 'multi'}.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
