#!/usr/bin/env python3
"""Experiment 0B: Self-Supervised Reliability Validity (methodology v9 S16.2).

For query q and estimator k:
    U_N_star(q)   = 1 - (1/N) * sum_i C(y_i)                  (from score_correctness.py)
    z_k^abs(q)    = 1 - | tilde U_k(q) - U_N_star(q) |          (evaluation-only, uses gold)
    z_k^SSL(q)    = eta*hat_a_k(q) + (1-eta)*hat_s_k(q)         (from ssl_signal.py, gold-free)

Primary test (v9 S16.2):
    rho_k^Spearman = SpearmanCorr( z_k^SSL(.), z_k^abs(.) )     across queries, per estimator k

Plus: per-query rank agreement between the SSL reliability ordering (rank
estimators by z_k^SSL, descending) and the absolute-calibration ordering
(rank by z_k^abs, descending) -- we report Spearman rank correlation of
the two per-query rankings, averaged over queries (a complementary view to
the across-query Spearman above, which correlates one estimator's SSL/abs
values across queries rather than comparing rankings within a query).

Confidence intervals: query-level bootstrap resampling (resample query
indices with replacement, B replicates, report the 2.5/97.5 percentiles).

v9 S16.2 also mandates a SEPARATE TruthfulQA-only breakdown ("consistency
can be high even when answers reproduce common false beliefs... A weak or
negative association is treated as evidence against the current SSL proxy,
not hidden as a limitation") -- pass --truthfulqa-ids (one query id per
line) to get that slice reported alongside the pooled result, never instead
of it.

This script is the one place in the pipeline that joins a gold-free signal
(z_k^SSL) against a gold-derived one (z_k^abs, via U_N_star) -- that is
its entire purpose; it is evaluation-only and never feeds back into
ssl_signal.py, bootstrap_stability.py, or any estimator computation.

Input:  ssl_signal.py output (z_k^SSL), calibrate_ecdf.py output (tilde U_k,
        reused here to build z_k^abs), and score_correctness.py output
        (U_N_star), matched by query id.
Output: JSON report: per-estimator rho_k^Spearman with bootstrap CI, mean
        per-query rank agreement, and the same broken out for the
        TruthfulQA-only subset if given.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

LOGGER = logging.getLogger("uetf.experiment_0b")


def load_keyed(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record:
                raise ValueError(f"{path}: line {line_no} missing 'id'.")
            records[str(record["id"])] = record
    return records


def load_id_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_joint_table(
    ssl_records: dict[str, dict[str, Any]],
    calibrated_records: dict[str, dict[str, Any]],
    correctness_records: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
    """Returns (query_ids, {estimator: {"ssl": [...], "abs": [...]}}) aligned
    by query id, dropping queries missing from any of the three inputs or
    with U_N_star=None (every response failed to score).
    """
    ids = sorted(set(ssl_records) & set(calibrated_records) & set(correctness_records))
    missing = (set(ssl_records) | set(calibrated_records) | set(correctness_records)) - set(ids)
    if missing:
        LOGGER.warning("Queries missing from at least one input, dropped: %s", sorted(missing))

    kept_ids: list[str] = []
    per_estimator: dict[str, dict[str, list[float]]] = {}
    for qid in ids:
        u_n_star = correctness_records[qid].get("U_N_star")
        if u_n_star is None:
            LOGGER.warning("id=%s has U_N_star=None (every response unscored) -- dropped.", qid)
            continue
        tilde_u = calibrated_records[qid]["calibrated"]
        z_ssl = ssl_records[qid]["z_k_SSL"]
        names = sorted(set(tilde_u) & set(z_ssl))
        kept_ids.append(qid)
        for name in names:
            z_abs = 1.0 - abs(tilde_u[name] - u_n_star)
            bucket = per_estimator.setdefault(name, {"ssl": [], "abs": []})
            bucket["ssl"].append(z_ssl[name])
            bucket["abs"].append(z_abs)
    return kept_ids, per_estimator


def spearman_with_bootstrap_ci(
    ssl_values: list[float], abs_values: list[float], *, n_boot: int, seed: int
) -> dict[str, float | None]:
    n = len(ssl_values)
    if n < 3:
        return {"rho": None, "p_value": None, "ci_low": None, "ci_high": None, "n": n}

    ssl_arr = np.asarray(ssl_values)
    abs_arr = np.asarray(abs_values)
    rho_obs, p_obs = stats.spearmanr(ssl_arr, abs_arr)

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(ssl_arr[idx], abs_arr[idx])
        if not np.isnan(r):
            boots.append(float(r))
    ci_low, ci_high = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
    return {"rho": float(rho_obs), "p_value": float(p_obs), "ci_low": ci_low, "ci_high": ci_high, "n": n}


def per_query_rank_agreement(
    ssl_records: dict[str, dict[str, Any]],
    calibrated_records: dict[str, dict[str, Any]],
    correctness_records: dict[str, dict[str, Any]],
    query_ids: list[str],
) -> dict[str, float | None]:
    """Mean Spearman rank correlation, per query, between the SSL ordering
    of estimators (by z_k^SSL, desc) and the absolute-calibration ordering
    (by z_k^abs, desc). Complements the across-query rho_k above.
    """
    agreements: list[float] = []
    for qid in query_ids:
        tilde_u = calibrated_records[qid]["calibrated"]
        z_ssl = ssl_records[qid]["z_k_SSL"]
        u_n_star = correctness_records[qid]["U_N_star"]
        names = sorted(set(tilde_u) & set(z_ssl))
        if len(names) < 3:
            continue
        ssl_rank = [z_ssl[n] for n in names]
        abs_rank = [1.0 - abs(tilde_u[n] - u_n_star) for n in names]
        r, _ = stats.spearmanr(ssl_rank, abs_rank)
        if not np.isnan(r):
            agreements.append(float(r))
    if not agreements:
        return {"mean": None, "std": None, "n_queries": 0}
    return {"mean": float(np.mean(agreements)), "std": float(np.std(agreements)), "n_queries": len(agreements)}


def run_slice(
    label: str,
    ssl_records: dict[str, dict[str, Any]],
    calibrated_records: dict[str, dict[str, Any]],
    correctness_records: dict[str, dict[str, Any]],
    *, n_boot: int, seed: int,
) -> dict[str, Any]:
    query_ids, per_estimator = build_joint_table(ssl_records, calibrated_records, correctness_records)
    per_estimator_results = {
        name: spearman_with_bootstrap_ci(vals["ssl"], vals["abs"], n_boot=n_boot, seed=seed)
        for name, vals in sorted(per_estimator.items())
    }
    rank_agreement = per_query_rank_agreement(ssl_records, calibrated_records, correctness_records, query_ids)
    return {
        "label": label,
        "n_queries": len(query_ids),
        "query_ids": query_ids,
        "per_estimator_rho_k_spearman": per_estimator_results,
        "per_query_rank_agreement": rank_agreement,
    }


def process(
    ssl_path: Path, calibrated_path: Path, correctness_path: Path, output_path: Path,
    *, truthfulqa_ids_path: Path | None, n_boot: int, seed: int,
    ids_paths: list[Path] | None = None,
) -> None:
    ssl_records = load_keyed(ssl_path)
    calibrated_records = load_keyed(calibrated_path)
    correctness_records = load_keyed(correctness_path)

    if ids_paths:
        # Restrict the analysis to an explicit id set (e.g. train+val; the TEST split is
        # never passed here). Applied to all three inputs before anything is computed.
        keep: set[str] = set()
        for path in ids_paths:
            keep |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
        LOGGER.info("Restricting analysis to %d ids from %s", len(keep), [str(p) for p in ids_paths])
        ssl_records = {k: v for k, v in ssl_records.items() if k in keep}
        calibrated_records = {k: v for k, v in calibrated_records.items() if k in keep}
        correctness_records = {k: v for k, v in correctness_records.items() if k in keep}

    report: dict[str, Any] = {
        "pooled": run_slice("pooled", ssl_records, calibrated_records, correctness_records, n_boot=n_boot, seed=seed)
    }

    truthfulqa_ids = load_id_list(truthfulqa_ids_path)
    if truthfulqa_ids is not None:
        tq_ssl = {k: v for k, v in ssl_records.items() if k in truthfulqa_ids}
        tq_cal = {k: v for k, v in calibrated_records.items() if k in truthfulqa_ids}
        tq_corr = {k: v for k, v in correctness_records.items() if k in truthfulqa_ids}
        if tq_ssl:
            report["truthfulqa_only"] = run_slice(
                "truthfulqa_only", tq_ssl, tq_cal, tq_corr, n_boot=n_boot, seed=seed
            )
        else:
            LOGGER.warning("--truthfulqa-ids given but none of those ids appear in the input files.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    pooled_rhos = {
        k: v["rho"] for k, v in report["pooled"]["per_estimator_rho_k_spearman"].items() if v["rho"] is not None
    }
    if pooled_rhos:
        best = max(pooled_rhos, key=pooled_rhos.get)
        worst = min(pooled_rhos, key=pooled_rhos.get)
        LOGGER.info(
            "Pooled n_queries=%d. Highest rho_k: %s=%.3f. Lowest rho_k: %s=%.3f. Full report: %s",
            report["pooled"]["n_queries"], best, pooled_rhos[best], worst, pooled_rhos[worst], output_path,
        )
    if "truthfulqa_only" in report:
        LOGGER.info("TruthfulQA-only n_queries=%d -- see report for per-estimator breakdown.", report["truthfulqa_only"]["n_queries"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssl", required=True, type=Path, help="ssl_signal.py output")
    parser.add_argument("--calibrated", required=True, type=Path, help="calibrate_ecdf.py output")
    parser.add_argument("--correctness", required=True, type=Path, help="score_correctness.py output")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--ids", type=Path, nargs="+", default=None,
                        help="Analysis id files (e.g. <cell>.train_ids.txt <cell>.val_ids.txt); never pass test ids")
    parser.add_argument("--truthfulqa-ids", type=Path, default=None, help="One query id per line (v9 S16.2 mandatory slice)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Query-level bootstrap replicates for CIs")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    process(
        args.ssl, args.calibrated, args.correctness, args.output,
        truthfulqa_ids_path=args.truthfulqa_ids, n_boot=args.n_boot, seed=args.seed,
        ids_paths=args.ids,
    )


if __name__ == "__main__":
    main()
