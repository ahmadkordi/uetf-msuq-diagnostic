#!/usr/bin/env python3
"""Brier score and ECE of the calibrated-uncertainty methods (DESCRIPTIVE; no selection is made from these numbers).

Reads the frozen test-split predictions saved by stats_extra.py (uby_<cell>.npz) and compares the predicted
uncertainty U(q) in [0,1] (e-CDF-normalised, so NOT a probability by construction) with the empirical error
rate U*(q) = 1 - acc_t(q). B3-Sum / B3-Min are raw, un-normalised scores and are excluded.
  Brier   = mean (U - U*)^2         (query level)
  ECE     = sum_b (n_b/n) |mean U_b - mean U*_b| over 10 equal-width bins of U
  skill   = 1 - Brier / Brier(constant = mean U* on the same split)   (the constant is an oracle reference: generous to the baseline)
Usage: python calibration_metrics.py --dir /mnt/output/experiment_0b/stats_extra
"""
import argparse, csv, glob
from pathlib import Path
import numpy as np

KEYS = {"B1": "B1", "B1@70": "B1at70", "B2": "B2", "B4": "B4", "B5": "B5", "B5'": "B5p", "B6": "B6"}


def ece(u, y, bins=10):
    idx = np.minimum((u * bins).astype(int), bins - 1); tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): tot += m.mean() * abs(u[m].mean() - y[m].mean())
    return float(tot)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, required=True); a = ap.parse_args(); rows = []
    for f in sorted(glob.glob(str(a.dir / "uby_*.npz"))):
        d = np.load(f); cell = Path(f).stem[4:]
        for thr in (90, 70):
            y = 1.0 - d[f"acc{thr}"]; const = float(np.mean((y.mean() - y) ** 2))
            for m, k in KEYS.items():
                u = np.clip(d[k], 0, 1); b = float(np.mean((u - y) ** 2))
                rows.append({"cell": cell, "thr": thr, "method": m, "brier": b, "brier_skill_vs_const": 1 - b / const, "ece": ece(u, y), "mean_U": float(u.mean()), "mean_Ustar": float(y.mean())})
    with open(a.dir / "calibration_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.dir / "calibration_metrics.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
