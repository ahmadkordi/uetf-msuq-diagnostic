#!/usr/bin/env python3
"""Diagnostic: why is s3_self_eval.U_p negatively correlated with the pool in
coqa_Llama2_13b and truthfulqa_Llama2_13b (Experiment 0A flag)?

Evaluation-only script (reads stage_h correctness, i.e. gold-derived labels) --
it is a diagnostic, never an input to Stage A-I.

For every cell it reports three groups of numbers so the three candidate
explanations can be told apart:

  (1) p_true distribution (saturation / degeneracy hypothesis)
      - frac of responses with p_true >= 0.9995, <= 0.0005, exactly 1.0, exactly 0.0
      - mean / std of p_true, mean within-query std (near 0 => the self-eval
        prompt returns the same confidence for all N responses of a query)
      - std and n_unique of the per-query U_p across queries
        (near-constant U_p => rank correlations are driven by ties/noise)

  (2) orientation vs. the rest of the pool (reproduces the 0A flag)
      - Spearman(U_p, mean-rank of the other 14 estimators)
      - Spearman(U_p, U_CP) and Spearman(U_p, s1_semantic.agreement.degree)

  (3) which side is "wrong" -- orientation vs. ground truth
      - Spearman(U_p, U_N_star) and Spearman(U_CP, U_N_star),
        U_N_star = 1 - mean correctness (higher = more wrong)
      If U_p is negatively correlated with the pool but POSITIVELY with
      U_N_star (and the pool is flat/negative with U_N_star), the self-eval
      estimator is the informative one and the pool is what misleads; if U_p
      is negative or ~0 with U_N_star too, U_p is simply not working in that
      cell. Either is a reportable finding -- not something to silently
      sign-flip away.

Inputs per cell (files under --dir):
    stage_b_s3_self_eval_<cell>.jsonl   (p_true list per query)
    stage_f_estimators_<cell>.jsonl     (raw estimator dict per query)
    stage_h_correctness_<cell>.jsonl    (U_N_star per query)
    stage_c_tensor_<cell>.jsonl         (U_CP, U_Tucker per query)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

LOGGER = logging.getLogger("uetf.diagnose_self_eval")

DEFAULT_CELLS = [
    f"{ds}_{m}"
    for ds in ("coqa", "hotpotqa", "nqopen", "truthfulqa")
    for m in ("Llama2_13b", "llama3.1_8b", "Q2.5_14b")
]
UP = "s3_self_eval.U_p"
SAT_HI, SAT_LO = 0.9995, 0.0005


def load_keyed(path: Path, keep: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[str(rec["id"])] = {k: rec[k] for k in keep if k in rec}
    return out


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r, _ = stats.spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def p_true_stats(s3: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    all_p: list[float] = []
    within_std: list[float] = []
    u_p: list[float] = []
    for rec in s3.values():
        p = np.asarray(rec["p_true"], dtype=np.float64)
        all_p.extend(p.tolist())
        within_std.append(float(p.std()))
        u_p.append(1.0 - float(p.mean()))
    p_arr = np.asarray(all_p)
    u_arr = np.asarray(u_p)
    return {
        "n_queries": len(s3),
        "n_responses": int(p_arr.size),
        "p_true_mean": float(p_arr.mean()),
        "p_true_std": float(p_arr.std()),
        "frac_ge_0.9995": float((p_arr >= SAT_HI).mean()),
        "frac_le_0.0005": float((p_arr <= SAT_LO).mean()),
        "frac_exactly_1": float((p_arr == 1.0).mean()),
        "frac_exactly_0": float((p_arr == 0.0).mean()),
        "mean_within_query_std": float(np.mean(within_std)),
        "U_p_std_across_queries": float(u_arr.std()),
        "U_p_n_unique": int(np.unique(np.round(u_arr, 6)).size),
    }


def analyze_cell(cell: str, base: Path) -> dict[str, Any]:
    s3 = load_keyed(base / f"stage_b_s3_self_eval_{cell}.jsonl", ("p_true",))
    sf = load_keyed(base / f"stage_f_estimators_{cell}.jsonl", ("estimators",))
    sh = load_keyed(base / f"stage_h_correctness_{cell}.jsonl", ("U_N_star",))
    # tensor scores live in stage_c_tensor (top-level U_CP / U_Tucker); merged
    # under the same names calibrate_ecdf.py uses ("tensor.U_CP", "tensor.U_Tucker").
    sc = load_keyed(base / f"stage_c_tensor_{cell}.jsonl", ("U_CP", "U_Tucker"))
    for qid, rec in sf.items():
        if qid in sc:
            rec["estimators"]["tensor.U_CP"] = sc[qid]["U_CP"]
            rec["estimators"]["tensor.U_Tucker"] = sc[qid]["U_Tucker"]

    ids = sorted(set(s3) & set(sf) & set(sh) & set(sc))
    ids = [i for i in ids if sh[i].get("U_N_star") is not None]
    result: dict[str, Any] = {"cell": cell, "n_joined": len(ids), "p_true": p_true_stats(s3)}

    est_names = sorted(sf[ids[0]]["estimators"])
    LOGGER.info("%s: estimator keys in stage_f: %s", cell, est_names)
    mat = np.asarray([[sf[i]["estimators"][n] for n in est_names] for i in ids], dtype=np.float64)
    col = {n: mat[:, j] for j, n in enumerate(est_names)}
    up_name = UP if UP in col else resolve(est_names, "u_p")
    if up_name is None:
        raise KeyError(f"{cell}: no U_p-like estimator among {est_names}")
    u_p = col[up_name]
    u_star = np.asarray([sh[i]["U_N_star"] for i in ids], dtype=np.float64)

    others = [n for n in est_names if n != up_name]
    # mean-rank of the other estimators = a simple, monotone-invariant pool summary
    rank_mean = np.mean([stats.rankdata(col[n]) for n in others], axis=0)

    cp_name = resolve(est_names, "u_cp") or resolve(est_names, "cp")
    deg_name = resolve(est_names, "s1", "agree", "degree") or resolve(est_names, "s1", "degree")
    LOGGER.info("%s: using U_p=%s, CP=%s, S1-degree=%s", cell, up_name, cp_name, deg_name)
    cp = col[cp_name] if cp_name else None
    result["orientation_vs_pool"] = {
        "spearman_Up_vs_pool_meanrank": safe_spearman(u_p, rank_mean),
        "spearman_Up_vs_U_CP": safe_spearman(u_p, cp) if cp is not None else None,
        "spearman_Up_vs_s1_agree_degree": safe_spearman(u_p, col[deg_name]) if deg_name else None,
    }
    result["vs_ground_truth_U_N_star"] = {
        "spearman_Up": safe_spearman(u_p, u_star),
        "spearman_U_CP": safe_spearman(cp, u_star) if cp is not None else None,
        "spearman_pool_meanrank": safe_spearman(rank_mean, u_star),
        "mean_U_N_star": float(u_star.mean()),
    }
    return result


def resolve(names: list[str], *must: str) -> str | None:
    """First estimator name containing all substrings (case-insensitive)."""
    hits = [n for n in names if all(m.lower() in n.lower() for m in must)]
    return hits[0] if hits else None


def fmt(x: float | None, nd: int = 3) -> str:
    return "  n/a" if x is None else f"{x:+.{nd}f}"


def print_report(rows: list[dict[str, Any]]) -> None:
    print("\n(1) p_true distribution")
    print(f"{'cell':26s} {'mean':>6s} {'std':>6s} {'>=.9995':>8s} {'<=.0005':>8s} {'==1':>6s} {'==0':>6s} {'inQstd':>7s} {'Up_std':>7s} {'Up_uniq':>8s}")
    for r in rows:
        p = r["p_true"]
        print(
            f"{r['cell']:26s} {p['p_true_mean']:6.3f} {p['p_true_std']:6.3f} "
            f"{p['frac_ge_0.9995']:8.3f} {p['frac_le_0.0005']:8.3f} {p['frac_exactly_1']:6.3f} "
            f"{p['frac_exactly_0']:6.3f} {p['mean_within_query_std']:7.3f} "
            f"{p['U_p_std_across_queries']:7.3f} {p['U_p_n_unique']:8d}"
        )
    print("\n(2) U_p orientation vs. the rest of the pool (Spearman)")
    print(f"{'cell':26s} {'vs pool':>9s} {'vs U_CP':>9s} {'vs deg(S1,agr)':>15s}")
    for r in rows:
        o = r["orientation_vs_pool"]
        print(
            f"{r['cell']:26s} {fmt(o['spearman_Up_vs_pool_meanrank']):>9s} "
            f"{fmt(o['spearman_Up_vs_U_CP']):>9s} {fmt(o['spearman_Up_vs_s1_agree_degree']):>15s}"
        )
    print("\n(3) Versus ground truth U_N_star (Spearman; higher U_N_star = more wrong)")
    print(f"{'cell':26s} {'U_p':>8s} {'U_CP':>8s} {'pool':>8s} {'mean U*':>8s}")
    for r in rows:
        g = r["vs_ground_truth_U_N_star"]
        print(
            f"{r['cell']:26s} {fmt(g['spearman_Up']):>8s} {fmt(g['spearman_U_CP']):>8s} "
            f"{fmt(g['spearman_pool_meanrank']):>8s} {g['mean_U_N_star']:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path, help="Directory holding the stage_b_s3 / stage_f / stage_h files")
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    rows = []
    for cell in args.cells:
        LOGGER.info("analyzing %s", cell)
        rows.append(analyze_cell(cell, args.dir))
    print_report(rows)
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        LOGGER.info("wrote %s", args.json_out)


if __name__ == "__main__":
    main()
