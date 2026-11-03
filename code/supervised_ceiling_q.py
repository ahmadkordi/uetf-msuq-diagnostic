#!/usr/bin/env python3
"""Supervised reference WITH vs WITHOUT query features (diagnostic, label-using, VAL split only; test untouched).

Feature sets (all standardised on TRAIN):
  U      the 15 calibrated estimator values
  U+Q    U plus phi_Q-lite (log(1+words), TTR, 7 first-word indicators)
  full   U, mu, sigma, range, DC_i and phi_Q-lite (the B6 input without z^SSL)
Models: logistic regression (C chosen on an internal 80/20 split of TRAIN) and histogram gradient boosting, as in
supervised_ceiling.py. Paired query-bootstrap CIs of: U+Q - U, full - U (the value of query/dispersion features),
and each of U, U+Q, full against B1 (best single estimator selected on TRAIN at the same threshold).
Usage (from /mnt/src):  python supervised_ceiling_q.py --data-dir /mnt/output --cells <cell> [<cell> ...] --n-boot 1000
Outputs: <data-dir>/experiment_0b/gate/ceilingQ_<cell>.csv  (one row per comparison)
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from gate_eval import Cell, paired_boot
from baseline_auroc import official_auroc
from supervised_ceiling import expand


def feats(C):
    U = C.U; disp = np.hstack([U.mean(1, keepdims=True), U.std(1, keepdims=True), U.max(1, keepdims=True) - U.min(1, keepdims=True),
                               np.abs(U[:, :, None] - U[:, None, :]).sum(2) / (C.K - 1)])
    return {"U": U, "U+Q": np.hstack([U, C.Q]), "full": np.hstack([U, disp, C.Q])}


def fit_predict(C, X, tr, va, thr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - mu) / sd; acc = C.acc[thr]
    rng = np.random.default_rng(0); perm = rng.permutation(len(tr)); cut = int(0.8 * len(tr)); a_i, b_i = tr[perm[:cut]], tr[perm[cut:]]
    best = (-1, 1.0)
    for Cc in (0.01, 0.1, 1.0, 10.0):
        Xe, y, w = expand(Xs[a_i], acc[a_i], C.N); m = LogisticRegression(C=Cc, max_iter=500).fit(Xe, y, sample_weight=w)
        s = C.auroc(m.predict_proba(Xs[b_i])[:, 1], b_i, thr)
        if s > best[0]: best = (s, Cc)
    Xe, y, w = expand(Xs[tr], acc[tr], C.N)
    lr = LogisticRegression(C=best[1], max_iter=500).fit(Xe, y, sample_weight=w)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, early_stopping=True, random_state=0).fit(Xe, y, sample_weight=w)
    return lr.predict_proba(Xs[va])[:, 1], gb.predict_proba(Xs[va])[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=1000); a = ap.parse_args()
    out = a.data_dir / "experiment_0b" / "gate"; out.mkdir(parents=True, exist_ok=True)
    for cell in a.cells:
        C = Cell(a.data_dir, cell, "cal"); tr, va = C.rows["train"], C.rows["val"]; F = feats(C); rows = []
        for thr in (90, 70):
            P = {}
            for fs, X in F.items():
                lr, gb = fit_predict(C, X, tr, va, thr); P[("LR", fs)] = lr; P[("GB", fs)] = gb
            b1 = max(range(C.K), key=lambda k: official_auroc(C.U[tr][:, k], C.acc[thr][tr], C.N)); P[("B1", "-")] = C.U[va][:, b1]
            cmp = [(("LR", "U+Q"), ("LR", "U")), (("LR", "full"), ("LR", "U")), (("GB", "U+Q"), ("GB", "U")), (("GB", "full"), ("GB", "U"))]
            cmp += [((m, fs), ("B1", "-")) for m in ("LR", "GB") for fs in ("U", "U+Q", "full")]
            for A, B in cmp:
                d = C.auroc(P[A], va, thr) - C.auroc(P[B], va, thr); lo, hi = paired_boot(C, va, thr, P[A], P[B], a.n_boot, 1)
                rows.append({"cell": cell, "thr": thr, "A": "/".join(A), "B": "/".join(B), "auroc_A": C.auroc(P[A], va, thr), "auroc_B": C.auroc(P[B], va, thr), "delta": d, "lo": lo, "hi": hi, "b1": C.names[b1]})
        with open(out / f"ceilingQ_{cell}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"[{cell}] done", flush=True)


if __name__ == "__main__":
    main()
