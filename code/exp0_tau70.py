#!/usr/bin/env python3
"""Experiment 0B decomposition at a different correctness threshold (default tau = 70).

EVALUATION-ONLY diagnostic, train+val ids only (never test). Nothing is fitted and
nothing is selected: the stored SSL components (ssl_<variant>_<cell>.jsonl), the stored
train-fit calibrated scores (stage_g_trainfit_<cell>.jsonl) and the stored judge scores
(stage_h_correctness_<cell>.jsonl) are read as they are. The only change against
exp0b_decompose.py is the error rate of each query:

    U_N_star_tau(q) = 1 - mean_i [ score_i(q) > tau ]

recomputed from the stored per-response judge scores. All statistics are computed by the
functions of exp0b_decompose.py itself (build_arrays, analyse), so the numbers are
directly comparable with Table 3 of the paper.

Writes only into a NEW folder (default <data-dir>/experiment_0b/tau70/); existing
files are never overwritten unless --force is given.

Checks printed for every cell:
  * stored U_N_star equals the value recomputed with threshold 90 (max abs difference);
  * with --check90: the tau = 90 decomposition recomputed here equals the stored
    exp0b_decomp_<variant>_<cell>.json (max abs difference of all reported means).

Run (from the code folder, venv active):
  python exp0_tau70.py --data-dir /mnt/output --check90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp0b_decompose as E  # noqa: E402  (same folder as the pipeline scripts)

DATASETS = ["coqa", "hotpotqa", "nqopen", "truthfulqa"]
BASE_MODELS = ["Llama2_13b", "llama3.1_8b", "Q2.5_14b"]


def u_star(rec: dict, tau: float):
    pr = rec.get("per_response") or []
    if not pr:
        return None
    return 1.0 - sum(1 for r in pr if r["score"] > tau) / len(pr)


def flatten(rep: dict) -> dict:
    """All numbers that the summary tables use, keyed by component."""
    out = {}
    for c, v in rep["components"].items():
        out[f"{c}|mean_rho"] = v["across_query"]["mean_rho"]
        out[f"{c}|within"] = v["within_query"]["mean"]
        out[f"{c}|between"] = v["between_query"]["rho"]
    out["n_queries"] = rep["n_queries"]
    return out


def max_diff(a: dict, b: dict) -> float:
    d = 0.0
    for k, x in a.items():
        y = b.get(k)
        if x is None or y is None:
            if (x is None) != (y is None):
                return float("inf")
            continue
        d = max(d, abs(float(x) - float(y)))
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output"))
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <data-dir>/experiment_0b/tau<T>")
    ap.add_argument("--threshold", type=float, default=70.0)
    ap.add_argument("--variant", default="cal", choices=["cal", "raw"])
    ap.add_argument("--cells", nargs="*", default=None, help="default: the 12 main cells")
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--check90", action="store_true", help="also recompute tau = 90 and compare with the stored decomposition")
    ap.add_argument("--force", action="store_true", help="allow overwriting files in the output folder")
    a = ap.parse_args()

    O = a.data_dir
    D = O / "experiment_0b"
    Sp = O / "experiment_0a" / "splits"
    out = a.out_dir or (D / f"tau{a.threshold:g}")
    out.mkdir(parents=True, exist_ok=True)
    cells = a.cells or [f"{d}_{m}" for m in BASE_MODELS for d in DATASETS]

    summary = {}
    for cell in cells:
        target = out / f"exp0b_decomp_{a.variant}_{cell}.json"
        if target.exists() and not a.force:
            sys.exit(f"{target} exists; use --force or another --out-dir")
        keep = E.read_ids([Sp / f"{cell}.train_ids.txt", Sp / f"{cell}.val_ids.txt"])   # never test
        ssl = E.load_keyed(D / f"ssl_{a.variant}_{cell}.jsonl")
        cal = E.load_keyed(D / f"stage_g_trainfit_{cell}.jsonl")
        corr = E.load_keyed(O / f"stage_h_correctness_{cell}.jsonl")

        # check 1: the stored U_N_star is the threshold-90 error rate
        ids = sorted(set(ssl) & set(cal) & set(corr) & keep)
        diffs = [abs(corr[i]["U_N_star"] - u_star(corr[i], 90.0)) for i in ids
                 if corr[i].get("U_N_star") is not None and u_star(corr[i], 90.0) is not None]
        chk_store = max(diffs) if diffs else float("nan")

        corr_t = {}
        for i, r in corr.items():
            u = u_star(r, a.threshold)
            corr_t[i] = dict(r, U_N_star=u)
        rep = E.analyse(*E.build_arrays(ssl, cal, corr_t, keep), a.etas, a.n_boot, a.seed)
        rep["threshold"] = a.threshold
        rep["frac_zero_correct"] = sum(1 for i in ids if corr_t[i]["U_N_star"] == 1.0) / max(1, len(ids))
        target.write_text(json.dumps(rep, indent=2))

        line = (f"{cell:24s} nQ={rep['n_queries']:5d}  stored-vs-recomputed(90) max|d|={chk_store:.2e}  "
                f"A within={rep['components']['z@eta=1']['within_query']['mean']:+.3f}  "
                f"B within={rep['components']['z@eta=0']['within_query']['mean']:+.3f}")
        if a.check90:
            stored = D / f"exp0b_decomp_{a.variant}_{cell}.json"
            if stored.exists():
                rep90 = E.analyse(*E.build_arrays(ssl, cal, corr, keep), a.etas, a.n_boot, a.seed)
                old = json.loads(stored.read_text())
                common = {k: v for k, v in flatten(rep90).items() if k in flatten(old)}
                line += f"  check90 max|d| vs stored={max_diff(common, flatten(old)):.2e}"
            else:
                line += "  check90: no stored file"
        print(line, flush=True)
        summary[cell] = {"n_queries": rep["n_queries"], "stored_vs_recomputed_90": chk_store,
                         "frac_zero_correct": rep["frac_zero_correct"]}
    (out / f"exp0b_decomp_{a.variant}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
