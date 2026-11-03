#!/usr/bin/env python3
"""KLE-style baseline (Nikitin et al., 2024) on the STORED, SYMMETRISED agreement matrix. NO FITTING, test split, official AUROC.

  W = matrix_agreement (continuous, symmetric, diagonal set to 1); D = diag(W 1)
  KLE-heat (primary): L = D - W (Nikitin et al. Sec. 3.1), K = expm(-t L), t = 0.3 (App. C default), kernel normalised as in Sec. 3.1
           (K_ij (K_ii K_jj)^-1/2 / N, unit trace), U = von Neumann entropy.
  KLE-heat-trace (sensitivity): same but plain trace normalisation (what the first run used); KLE-heat-normL: normalised Laplacian (App. C).
Deviations to state in the paper: (i) the original builds W from one-hot NLI predictions with weights (1, 0.5, 0) for
entailment/neutral/contradiction; we use the stored continuous agreement matrix (symmetrised mean of NLI probabilities);
(ii) t = 0.3 is the heat-kernel default as read from the paper, not tuned; (iii) Laplacian/normalisation re-checked against the PDF text after a first run that used the wrong primary choice (normalised L, trace-only), which gave AUROC<0.5; (iv) same NLI source as the pool, not independent.
Usage (from /mnt/src):  python kle_baseline.py --data-dir /mnt/output --cells <cell> ... --n-boot 1000
Output: <data-dir>/experiment_0b/gate/kle_<cell>.csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from gate_eval import Cell
from baseline_auroc import official_auroc

T = 0.3


def vne(W, normalised, unit_diag=True, t=T):
    """VNE of the heat kernel. unit_diag=True: paper normalisation K <- K_ij (K_ii K_jj)^-1/2 / N (Nikitin Sec. 3.1);
    unit_diag=False: plain trace normalisation (sensitivity only)."""
    W = np.array(W, float); W = (W + W.T) / 2; np.fill_diagonal(W, 1.0); W = np.clip(W, 0, None)
    d = W.sum(1)
    L = np.eye(len(W)) - W / np.sqrt(np.outer(d, d)) if normalised else np.diag(d) - W
    lam, V = np.linalg.eigh((L + L.T) / 2)
    K = (V * np.exp(-t * lam)) @ V.T
    if unit_diag:
        dk = np.sqrt(np.diag(K)); K = K / np.outer(dk, dk)
    K = K / np.trace(K)
    ev = np.linalg.eigvalsh((K + K.T) / 2); ev = ev[ev > 1e-12]
    return float(-(ev * np.log(ev)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=1000); a = ap.parse_args()
    O = a.data_dir; out = O / "experiment_0b" / "gate"
    for cell in a.cells:
        C = Cell(O, cell, "cal"); te, tr = C.rows["test"], C.rows["train"]; ids = C.split["test"]; recs = {}
        with open(O / f"stage_b_s1_semantic_{cell}.jsonl", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln); recs[str(r["id"])] = r["matrix_agreement"]
        U = {"KLE-heat": np.array([vne(recs[i], False, True) for i in ids]),            # primary: L=D-W, paper normalisation
             "KLE-heat-trace": np.array([vne(recs[i], False, False) for i in ids]),         # sensitivity: L=D-W, trace-only
             "KLE-heat-normL": np.array([vne(recs[i], True, True) for i in ids])}           # sensitivity: normalised Laplacian (App. C)
        b1 = {thr: max(range(C.K), key=lambda k: official_auroc(C.U[tr][:, k], C.acc[thr][tr], C.N)) for thr in (90, 70)}
        ref = {"B2": C.U[te].mean(1), "B3-Sum": (C.rawTK + C.rawCP)[te]}
        rng = np.random.default_rng(1); n = len(te); idxs = [rng.integers(0, n, n) for _ in range(a.n_boot)]; rows = []
        for thr in (90, 70):
            acc = C.acc[thr][te]; ref_t = dict(ref); ref_t["B1"] = C.U[te][:, b1[thr]]
            for m, u in U.items():
                row = {"cell": cell, "thr": thr, "method": m, "auroc": official_auroc(u, acc, C.N)}
                bs = np.array([official_auroc(u[i], acc[i], C.N) for i in idxs]); row["lo"], row["hi"] = np.percentile(bs, [2.5, 97.5])
                for rname, ru in ref_t.items():
                    row[f"d_{rname}"] = row["auroc"] - official_auroc(ru, acc, C.N)
                    bd = bs - np.array([official_auroc(ru[i], acc[i], C.N) for i in idxs])
                    row[f"lo_{rname}"], row[f"hi_{rname}"] = np.percentile(bd, [2.5, 97.5])
                rows.append(row)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("cell", "thr", "method", "auroc"), k))
        with open(out / f"kle_{cell}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"[{cell}] done", flush=True)


if __name__ == "__main__":
    main()
