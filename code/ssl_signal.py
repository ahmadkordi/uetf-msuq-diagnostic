#!/usr/bin/env python3
"""Stage I.4: Self-supervised pseudo-reliability signal z_k^SSL for U-ETF.

Methodology v9 S11.1, S11.2, S11.4:

    Signal A (cross-estimator agreement), from calibrated estimators tilde U:
        m_{-k}(q)   = median_{j != k} tilde U_j(q)
        d_cons_k(q) = |tilde U_k(q) - m_{-k}(q)|
        a_k(q)      = exp(-d_cons_k(q) / sigma_cons)          in (0, 1]

    Signal B (bootstrap sampling stability), from bootstrap_stability.py:
        s_k(q) = exp(-d_stab_k(q) / sigma_stab)                in (0, 1]
        (already computed upstream; this script only consumes s_k.)

    Combination:
        hat_a_k(q) = Norm_k(a_k(q)),  hat_s_k(q) = Norm_k(s_k(q))
        z_k^SSL(q) = eta * hat_a_k(q) + (1 - eta) * hat_s_k(q),   0 <= eta <= 1
        zbar_k^SSL(q) = (z_k^SSL(q) + alpha/K) / (sum_j z_j^SSL(q) + alpha)

Phase 0 / design-choice audit notes (v9 leaves these underspecified by
design -- it is a draft methodology, not yet a frozen spec; each choice
below is flagged rather than silently picked):

  - "Norm_k(.)" is never pinned down to a formula in the draft. We implement
    it as MIN-MAX normalization across the K estimators within one query:
    Norm_k(x_k) = (x_k - min_j x_j) / (max_j x_j - min_j x_j + eps), which
    keeps a_k/s_k in the same comparable [0,1] range the rest of the
    pipeline uses. If a degenerate query has all-equal a_k (or s_k) across
    estimators, we fall back to a uniform 1/K vector rather than dividing by
    zero -- document any such query if it appears in real data.
  - sigma_cons (the Signal-A scale) is "fitted from the training
    distribution and frozen thereafter" (S11.1) exactly like sigma_stab in
    bootstrap_stability.py. This script mirrors that workflow: omit
    --sigma-cons when fitting on a training split (a candidate value is
    logged), pass the frozen value for any split being scored for real.
  - eta (Signal A vs Signal B mixing weight) and alpha (Dirichlet shrinkage
    strength) are NOT given default values anywhere in the v9 draft. We
    default eta=0.5 (equal weight, the least assumption-laden choice) and
    alpha=1.0 (mild shrinkage, one "pseudo-estimator" of weight), but both
    are explicit CLI flags specifically so this default is never silently
    load-bearing in a reported result -- sweep them and report sensitivity
    before trusting a specific number.

This module is deliberately independent of correctness labels and gold
answers -- it only ever consumes Stage G (e-CDF calibrated) and
bootstrap_stability.py outputs, neither of which touches gold_answer/
best_answer.

Input:  calibrate_ecdf.py output (for tilde U_k) and bootstrap_stability.py
        output (for s_k), matched by query id.
Output: JSONL with, per query, a_k, s_k, hat_a_k, hat_s_k, z_k^SSL, and
        zbar_k^SSL for every estimator k.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.ssl_signal")
_EPS = 1e-8


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


def leave_one_out_consensus(tilde_u: dict[str, float]) -> dict[str, float]:
    """m_{-k}(q) = median_{j != k} tilde U_j(q), for every k."""
    names = sorted(tilde_u)
    values = np.asarray([tilde_u[n] for n in names], dtype=np.float64)
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        others = np.delete(values, i)
        out[name] = float(np.median(others)) if len(others) else float(values[i])
    return out


def min_max_norm(values: dict[str, float]) -> dict[str, float]:
    """Documented Norm_k choice: min-max across estimators within one query."""
    names = list(values)
    arr = np.asarray([values[n] for n in names], dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < _EPS:
        uniform = 1.0 / len(names)
        return {n: uniform for n in names}
    return {n: float((v - lo) / (hi - lo)) for n, v in zip(names, arr)}


def compute_signal_a(calibrated: dict[str, float], sigma_cons: float) -> dict[str, float]:
    consensus = leave_one_out_consensus(calibrated)
    return {
        name: float(np.exp(-abs(calibrated[name] - consensus[name]) / sigma_cons))
        for name in calibrated
    }


def combine_query(
    calibrated: dict[str, float],
    s_k: dict[str, float],
    *,
    sigma_cons: float,
    eta: float,
    alpha: float,
) -> dict[str, Any]:
    names = sorted(set(calibrated) & set(s_k))
    missing = set(calibrated) ^ set(s_k)
    if missing:
        LOGGER.warning("Estimator name mismatch between calibrated/s_k, ignored: %s", sorted(missing))

    a_k = compute_signal_a({n: calibrated[n] for n in names}, sigma_cons)
    s_k_sub = {n: s_k[n] for n in names}
    hat_a = min_max_norm(a_k)
    hat_s = min_max_norm(s_k_sub)

    z_ssl = {n: float(eta * hat_a[n] + (1.0 - eta) * hat_s[n]) for n in names}
    total = sum(z_ssl.values()) + alpha
    k = len(names)
    z_bar = {n: float((z_ssl[n] + alpha / k) / total) for n in names} if k else {}

    return {
        "a_k": a_k, "s_k": s_k_sub, "hat_a_k": hat_a, "hat_s_k": hat_s,
        "z_k_SSL": z_ssl, "zbar_k_SSL": z_bar,
    }


def process_file(
    calibrated_path: Path, bootstrap_path: Path, output_path: Path,
    *, sigma_cons: float | None, eta: float, alpha: float, limit: int | None,
    train_ids: set[str] | None = None,
) -> None:
    calibrated_records = load_keyed(calibrated_path)
    bootstrap_records = load_keyed(bootstrap_path)
    ids = [i for i in calibrated_records if i in bootstrap_records]
    missing = set(calibrated_records) ^ set(bootstrap_records)
    if missing:
        LOGGER.warning("IDs present in only one input file, skipped: %s", sorted(missing))

    # If sigma_cons is not frozen yet, compute d_cons for every query/estimator
    # first (one pass) so we can log a candidate fitted value -- same
    # two-step workflow as bootstrap_stability.py's --sigma-stab.
    if sigma_cons is None:
        pooled: list[float] = []
        fit_ids = ids
        if train_ids is not None:
            fit_ids = [i for i in ids if i in train_ids]
            if not fit_ids:
                raise ValueError("--train-ids given but none of those ids are present in the inputs.")
            LOGGER.info("Fitting sigma_cons on %d TRAIN queries (of %d).", len(fit_ids), len(ids))
        else:
            LOGGER.warning("No --train-ids: sigma_cons is fit on ALL %d queries in the input (not train-only).", len(ids))
        for qid in fit_ids:
            tilde_u = calibrated_records[qid]["calibrated"]
            consensus = leave_one_out_consensus(tilde_u)
            pooled.extend(abs(tilde_u[n] - consensus[n]) for n in tilde_u)
        fitted = float(statistics.median(pooled)) if pooled else 1.0
        LOGGER.warning(
            "--sigma-cons not given. If these queries are your TRAINING split, a candidate "
            "frozen sigma_cons (median d_cons pooled across all estimators/queries) is "
            "%.6f -- freeze and reuse it for eval/test via --sigma-cons, per methodology "
            "v9 S11.1. Using this candidate value for THIS run.",
            fitted,
        )
        sigma_cons = fitted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for qid in ids:
            if limit is not None and processed >= limit:
                break
            tilde_u = calibrated_records[qid]["calibrated"]
            s_k = bootstrap_records[qid].get("s_k")
            if s_k is None:
                raise ValueError(
                    f"id={qid}: bootstrap_stability.py output has no 's_k' -- re-run that "
                    "script with --sigma-stab set (frozen value) before computing z_k^SSL."
                )
            combined = combine_query(tilde_u, s_k, sigma_cons=sigma_cons, eta=eta, alpha=alpha)
            result = {
                "id": qid,
                "question": calibrated_records[qid].get("question", ""),
                "params": {"sigma_cons": sigma_cons, "eta": eta, "alpha": alpha},
                **combined,
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrated", required=True, type=Path, help="calibrate_ecdf.py output")
    parser.add_argument("--bootstrap", required=True, type=Path, help="bootstrap_stability.py output (must have s_k)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--sigma-cons", type=float, default=None,
        help="Frozen scale for a_k = exp(-d_cons/sigma_cons). Omit to fit-and-use a candidate "
             "value on this run (log it, freeze it, reuse via this flag on eval/test).",
    )
    parser.add_argument("--eta", type=float, default=0.5, help="Signal-A/Signal-B mixing weight (v9 leaves this unset; see docstring)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet shrinkage strength (v9 leaves this unset; see docstring)")
    parser.add_argument("--train-ids", type=Path, default=None,
                        help="Text file, one id per line. When sigma_cons is not given it is fit on these ids only.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if not 0.0 <= args.eta <= 1.0:
        raise ValueError("--eta must be in [0, 1].")
    if args.alpha < 0:
        raise ValueError("--alpha must be >= 0.")
    process_file(
        args.calibrated, args.bootstrap, args.output,
        sigma_cons=args.sigma_cons, eta=args.eta, alpha=args.alpha, limit=args.limit,
        train_ids=(
            {line.strip() for line in args.train_ids.read_text().splitlines() if line.strip()}
            if args.train_ids is not None else None
        ),
    )


if __name__ == "__main__":
    main()
