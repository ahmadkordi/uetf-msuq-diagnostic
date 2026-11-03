#!/usr/bin/env python3
"""Post-hoc: Spearman correlation on the TEST queries between KLE-heat (final spec of kle_baseline.py), B3-Sum and B2 (mean of the pool),
plus Spearman between B3-Sum and the semantic-only estimator pool mean. Descriptive only, nothing is selected.
Usage (from /mnt/src): python kle_spearman.py --data-dir /mnt/output --cells $CELLS
Output: prints a table and writes <data-dir>/experiment_0b/gate/kle_spearman.csv"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from gate_eval import Cell
from kle_baseline import vne

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
a = ap.parse_args(); O = a.data_dir; rows = []
for cell in a.cells:
    C = Cell(O, cell, "cal"); te = C.rows["test"]; ids = C.split["test"]; recs = {}
    with open(O / f"stage_b_s1_semantic_{cell}.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln); recs[str(r["id"])] = r["matrix_agreement"]
    kle = np.array([vne(recs[i], False, True) for i in ids]); b3 = (C.rawTK + C.rawCP)[te]; b2 = C.U[te].mean(1)
    r = {"cell": cell, "n": len(ids), "rho_KLE_B3Sum": spearmanr(kle, b3)[0], "rho_KLE_B2": spearmanr(kle, b2)[0], "rho_B3Sum_B2": spearmanr(b3, b2)[0]}
    rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
print("median rho(KLE,B3-Sum) =", round(float(np.median([r["rho_KLE_B3Sum"] for r in rows])), 3))
with open(O / "experiment_0b" / "gate" / "kle_spearman.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
