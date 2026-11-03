#!/usr/bin/env python3
"""Summarize experiment_0b.py JSON reports across cells into one table per variant.

Reads <dir>/exp0b_<variant>_<tag>_<cell>.json for every cell and prints, per variant,
rho_k^Spearman (z_SSL vs z_abs, across queries) with '*' when the bootstrap 95% CI excludes 0,
plus the mean per-query rank agreement. Also writes a CSV (--csv-out) for the paper tables.
Evaluation-only (reads outputs that used gold-derived U_N_star).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CELLS = [f"{ds}_{m}" for ds in ("coqa", "hotpotqa", "nqopen", "truthfulqa")
         for m in ("Llama2_13b", "llama3.1_8b", "Q2.5_14b")]


def short(name: str) -> str:
    return (name.replace("s1_semantic", "S1").replace("s2_claim", "S2").replace("s3_self_eval", "S3")
            .replace("agreement", "agr").replace("disagreement", "dis").replace("eccentricity", "ecc")
            .replace("tensor.", "T."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("/mnt/output/experiment_0b"))
    ap.add_argument("--tag", default="trainval")
    ap.add_argument("--variants", nargs="+", default=["raw", "cal"])
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()
    rows_csv = []
    for variant in args.variants:
        reports = {}
        for cell in CELLS:
            path = args.dir / f"exp0b_{variant}_{args.tag}_{cell}.json"
            if path.is_file():
                reports[cell] = json.loads(path.read_text())
        if not reports:
            print(f"[{variant}] no reports found"); continue
        names = sorted({n for r in reports.values() for n in r["pooled"]["per_estimator_rho_k_spearman"]})
        print(f"\n=== variant={variant} tag={args.tag}  rho_k (Spearman z_SSL vs z_abs); * = 95% CI excludes 0")
        print(f"{'cell':24s}{'n':>6s} " + " ".join(f"{short(n)[:11]:>11s}" for n in names) + "   rankAgr")
        for cell, rep in reports.items():
            pe = rep["pooled"]["per_estimator_rho_k_spearman"]
            cells_txt = []
            for n in names:
                e = pe.get(n)
                if not e or e["rho"] is None:
                    cells_txt.append(f"{'n/a':>11s}")
                else:
                    star = "*" if (e["ci_low"] is not None and (e["ci_low"] > 0 or e["ci_high"] < 0)) else " "
                    cells_txt.append(f"{e['rho']:+10.3f}{star}")
                    rows_csv.append([variant, cell, n, e["rho"], e["ci_low"], e["ci_high"], e["p_value"], e["n"]])
            ra = rep["pooled"]["per_query_rank_agreement"]["mean"]
            print(f"{cell:24s}{rep['pooled']['n_queries']:6d} " + " ".join(cells_txt)
                  + f"   {('n/a' if ra is None else f'{ra:+.3f}')}")
    if args.csv_out:
        with args.csv_out.open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["variant", "cell", "estimator", "rho", "ci_low", "ci_high", "p_value", "n"])
            w.writerows(rows_csv)
        print(f"\nCSV -> {args.csv_out}")


if __name__ == "__main__":
    main()
