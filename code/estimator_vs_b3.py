#!/usr/bin/env python3
"""Direct check of whether the claim source / tensor add anything: TEST AUROC of every single estimator of the pool (no fitting, official AUROC),
with the semantic-only graph estimators (S1.agr.*, S1.dis.*) and paired bootstrap difference to B3-Sum, and the mean of the 3 S1.agr estimators.
Usage (from /mnt/src): python estimator_vs_b3.py --data-dir /mnt/output --cells $CELLS --n-boot 1000
Output: <data-dir>/experiment_0b/gate/estb3_<cell>.csv (cell, thr, name, auroc, d_B3Sum, lo, hi)"""
import argparse, csv
from pathlib import Path
import numpy as np
from gate_eval import Cell
from baseline_auroc import official_auroc

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
ap.add_argument("--n-boot", type=int, default=1000); a = ap.parse_args(); O = a.data_dir
for cell in a.cells:
    C = Cell(O, cell, "cal"); te = C.rows["test"]; rng = np.random.default_rng(1); n = len(te); idxs = [rng.integers(0, n, n) for _ in range(a.n_boot)]
    b3 = (C.rawTK + C.rawCP)[te]; cand = {nm: C.U[te][:, k] for k, nm in enumerate(C.names)}
    s1a = [k for k, nm in enumerate(C.names) if nm.startswith("s1_semantic.agreement")]
    cand["S1.agr.mean3"] = C.U[te][:, s1a].mean(1); cand["B2(pool mean)"] = C.U[te].mean(1); cand["B3-Sum"] = b3
    rows = []
    for thr in (90, 70):
        acc = C.acc[thr][te]; ref = np.array([official_auroc(b3[i], acc[i], C.N) for i in idxs]); a0 = official_auroc(b3, acc, C.N)
        for nm, u in cand.items():
            au = official_auroc(u, acc, C.N); bs = np.array([official_auroc(u[i], acc[i], C.N) for i in idxs]) - ref
            lo, hi = np.percentile(bs, [2.5, 97.5]); rows.append({"cell": cell, "thr": thr, "name": nm, "auroc": au, "d_B3Sum": au - a0, "lo": lo, "hi": hi})
    with open(O / "experiment_0b" / "gate" / f"estb3_{cell}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[{cell}] done", flush=True)
