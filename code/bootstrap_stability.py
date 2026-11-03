#!/usr/bin/env python3
"""Stage I.2: Bootstrap sampling-stability signal (Signal B) for U-ETF's SSL gate.

Methodology v9 §11.2 / §15 (bootstrap control):
    For bootstrap replicate b = 1..B_s, resample the N responses of a query
    WITH REPLACEMENT at full size N (never a smaller sub-sample -- v9 is
    explicit that shrinking N would confound "resampling sensitivity" with
    "small-sample noise"):

        Y^(b)(q) ~ Bootstrap(Y(q), N),   |Y^(b)(q)| = N.

    Recompute every estimator k on each Y^(b)(q), then:

        d_stab_k(q) = MAD_b( U_k^(b)(q) )                  (median abs. deviation)
        s_k(q)      = exp( -d_stab_k(q) / sigma_stab )      in (0, 1]

    sigma_stab is "fitted from the training distribution and frozen
    thereafter" (§11.2) -- this script supports that two-step workflow via
    --fit-sigma (prints the fitted scale; run once on a training split) and
    --sigma-stab (consumes a frozen value on any split, train or eval).

Implementation note -- no model re-querying required:
    Every estimator in this pipeline (graph EigV/Degree/Eccentricity, tensor
    U_CP/U_Tucker, self-eval scalar U_p) is a function of (a) the NxN
    pairwise similarity matrices already computed in Stage B (semantic_nli.py
    / claim_nli.py) and (b) the per-response p_true vector already computed
    in Stage B.4 (self_eval.py). Resampling the response LIST with
    replacement and recomputing these matrices is therefore equivalent to
    indexing the ALREADY-COMPUTED matrix with the resampled index list
    (W_b = W[idx][:, idx]) -- no NLI/self-eval model call is re-run. This is
    exact, not an approximation, for the graph/tensor estimators: W_b is
    exactly the pairwise-similarity matrix of the resampled response list,
    since if response i is resampled into two positions a and b, W_b[a, b] =
    W[i, i] = 1.0, which is the correct entailment score for a response
    compared with an identical copy of itself.

    One documented approximation: self-eval p_true_i was computed in Stage
    B.4 using the ORIGINAL full set of other responses as brainstormed
    "other answers" context. Resampling p_true_b(q) = p_true[idx] does NOT
    re-run the self-eval model with the resampled brainstorm context -- that
    would require a fresh forward pass per bootstrap replicate. We reuse the
    already-computed per-response p_true scalar and only resample WHICH
    responses are included/duplicated. This is a deliberate compute-cost
    trade-off, not a silent one: U_p's bootstrap instability therefore
    reflects sensitivity to "which responses are present," not sensitivity
    to "what the brainstorm context looked like."

This module is deliberately independent of correctness labels and gold
answers.

Input:  Stage-B S^(1) (semantic_nli.py), S^(2) (claim_nli.py), and
        optionally S^(3) (self_eval.py) outputs, matched by query id.
Output: JSONL with, per query, every estimator's raw bootstrap replicate
        values, d_stab_k, and (if --sigma-stab given) s_k.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.bootstrap_stability")

# Re-used, not reimplemented -- see graph_estimators.py / tensor_decompose.py
# for the audited definitions of every estimator computed below.
from graph_estimators import estimators_for_matrix  # noqa: E402
from tensor_decompose import (  # noqa: E402
    build_cp_tensor,
    build_tucker_tensor,
    cp_rank_sweep,
    tucker_rank_sweep,
)


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


def mad(values: list[float]) -> float:
    """Median absolute deviation (matches methodology's MAD_b)."""
    if not values:
        return 0.0
    med = statistics.median(values)
    return float(statistics.median(abs(v - med) for v in values))


def resample_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def compute_replicate_estimators(
    s1_agree: np.ndarray,
    s1_disagree: np.ndarray,
    s2_agree: np.ndarray,
    s2_disagree: np.ndarray,
    p_true: np.ndarray | None,
    idx: np.ndarray,
    *,
    cp_rank_range: range,
    tucker_rank_range: range,
    tucker_third_mode_rank: int,
    seed: int,
) -> dict[str, float | None]:
    """Recompute every estimator on one bootstrap-resampled response set."""
    ix = np.ix_(idx, idx)
    s1a_b, s1d_b = s1_agree[ix], s1_disagree[ix]
    s2a_b, s2d_b = s2_agree[ix], s2_disagree[ix]

    out: dict[str, float] = {}
    for source_name, agree_b, disagree_b in (("s1_semantic", s1a_b, s1d_b), ("s2_claim", s2a_b, s2d_b)):
        for variant, w in (("agreement", agree_b), ("disagreement", disagree_b)):
            for name, value in estimators_for_matrix(w).items():
                out[f"{source_name}.{variant}.{name}"] = value

    if p_true is not None:
        p_b = p_true[idx]
        out["s3_self_eval.U_p"] = float(1.0 - np.mean(p_b))

    s1_dict_b = {"matrix_agreement": s1a_b, "matrix_disagreement": s1d_b}
    s2_dict_b = {"matrix_agreement": s2a_b, "matrix_disagreement": s2d_b}
    # Bootstrap-with-replacement duplicates responses (about 12.6 distinct of 20 in
    # expectation), which makes some ALS normal-equation matrices exactly singular
    # (numpy.linalg.LinAlgError inside tensorly.parafac / tucker). Observed on real
    # data (3 of 156 shards crashed). Such a replicate is recorded as FAILED (None)
    # for that tensor estimator only -- never silently replaced -- and counted, so
    # the failure rate is reported. Graph estimators and U_p are unaffected.
    try:
        cp_tensor_b = build_cp_tensor(s1_dict_b, s2_dict_b)
        cp_results = cp_rank_sweep(cp_tensor_b, cp_rank_range, seed=seed)
        out["tensor.U_CP"] = float(sum(r["uncertainty"] for r in cp_results))
    except np.linalg.LinAlgError as exc:
        LOGGER.warning("CP replicate failed (singular ALS solve): %s", exc)
        out["tensor.U_CP"] = None
    try:
        tucker_tensor_b = build_tucker_tensor(s1_dict_b, s2_dict_b)
        tucker_results = tucker_rank_sweep(
            tucker_tensor_b, tucker_rank_range, third_mode_rank=tucker_third_mode_rank, seed=seed
        )
        out["tensor.U_Tucker"] = float(sum(r["uncertainty"] for r in tucker_results))
    except np.linalg.LinAlgError as exc:
        LOGGER.warning("Tucker replicate failed (singular solve): %s", exc)
        out["tensor.U_Tucker"] = None
    return out


def process_query(
    s1: dict[str, Any],
    s2: dict[str, Any],
    s3: dict[str, Any] | None,
    *,
    n_bootstrap: int,
    seed: int,
    cp_rank_range: range,
    tucker_rank_range: range,
    tucker_third_mode_rank: int,
    sigma_stab: float | None,
) -> dict[str, Any]:
    s1_agree = np.asarray(s1["matrix_agreement"], dtype=np.float64)
    s1_disagree = np.asarray(s1["matrix_disagreement"], dtype=np.float64)
    s2_agree = np.asarray(s2["matrix_agreement"], dtype=np.float64)
    s2_disagree = np.asarray(s2["matrix_disagreement"], dtype=np.float64)
    n = s1_agree.shape[0]
    p_true = np.asarray(s3["p_true"], dtype=np.float64) if s3 is not None else None

    rng = np.random.default_rng(seed)
    replicate_values: dict[str, list[float]] = {}
    for b in range(n_bootstrap):
        idx = resample_indices(n, rng)
        values = compute_replicate_estimators(
            s1_agree, s1_disagree, s2_agree, s2_disagree, p_true, idx,
            cp_rank_range=cp_rank_range, tucker_rank_range=tucker_rank_range,
            tucker_third_mode_rank=tucker_third_mode_rank, seed=seed + b,
        )
        for name, value in values.items():
            replicate_values.setdefault(name, []).append(value)

    MIN_VALID = 3  # MAD over fewer than 3 replicates is not meaningful
    n_valid = {name: sum(v is not None for v in values) for name, values in replicate_values.items()}
    d_stab: dict[str, float | None] = {}
    for name, values in replicate_values.items():
        valid = [v for v in values if v is not None]
        d_stab[name] = mad(valid) if len(valid) >= MIN_VALID else None
    result: dict[str, Any] = {
        "n_bootstrap": n_bootstrap,
        "replicates": replicate_values,
        "n_valid_replicates": n_valid,
        "d_stab": d_stab,
    }
    if sigma_stab is not None:
        result["s_k"] = {
            name: (None if d is None else float(np.exp(-d / sigma_stab))) for name, d in d_stab.items()
        }
        result["sigma_stab"] = sigma_stab
    return result


def process_file(
    s1_path: Path, s2_path: Path, s3_path: Path | None, output_path: Path,
    *, n_bootstrap: int, seed: int, cp_rank_range: range, tucker_rank_range: range,
    tucker_third_mode_rank: int, sigma_stab: float | None, limit: int | None,
    shard_index: int = 0, num_shards: int = 1,
) -> None:
    s1_records = load_keyed(s1_path)
    s2_records = load_keyed(s2_path)
    s3_records = load_keyed(s3_path) if s3_path is not None else {}

    ids = [i for i in s1_records if i in s2_records]
    missing = set(s1_records) ^ set(s2_records)
    if missing:
        LOGGER.warning("IDs present in only one of S1/S2, skipped: %s", sorted(missing))

    if num_shards > 1:
        # Deterministic split of the query list so independent processes can each
        # take a disjoint slice. Per-query results do not depend on the shard
        # (the bootstrap RNG is seeded per query), so sharding changes nothing numerically.
        ids = ids[shard_index::num_shards]
        LOGGER.info("shard %d/%d -> %d queries", shard_index, num_shards, len(ids))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    all_d_stab: dict[str, list[float]] = {}
    with output_path.open("w", encoding="utf-8") as dst:
        for query_id in ids:
            if limit is not None and processed >= limit:
                break
            s1, s2 = s1_records[query_id], s2_records[query_id]
            s3 = s3_records.get(query_id)
            per_query = process_query(
                s1, s2, s3,
                n_bootstrap=n_bootstrap, seed=seed,
                cp_rank_range=cp_rank_range, tucker_rank_range=tucker_rank_range,
                tucker_third_mode_rank=tucker_third_mode_rank, sigma_stab=sigma_stab,
            )
            for name, d in per_query["d_stab"].items():
                if d is not None:
                    all_d_stab.setdefault(name, []).append(d)

            result = {"id": query_id, "question": s1.get("question", s2.get("question", "")), **per_query}
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            LOGGER.info("id=%s n_bootstrap=%d estimators=%d", query_id, n_bootstrap, len(per_query["d_stab"]))

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)
    if sigma_stab is None:
        pooled = sorted(v for values in all_d_stab.values() for v in values)
        if pooled:
            fitted = float(statistics.median(pooled))
            LOGGER.warning(
                "--sigma-stab not given -- no s_k written. If this file's queries are your "
                "TRAINING split, a candidate frozen sigma_stab (median d_stab pooled across "
                "all estimators/queries) is %.6f -- freeze and reuse it for eval/test via "
                "--sigma-stab, per methodology v9 S11.2.",
                fitted,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s1", required=True, type=Path, help="semantic_nli.py output")
    parser.add_argument("--s2", required=True, type=Path, help="claim_nli.py output")
    parser.add_argument("--s3", type=Path, default=None, help="self_eval.py output (optional, adds U_p stability)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10, help="B_s, full-size bootstrap replicates (v9 default: 10)")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--cp-rank-min", type=int, default=10)
    parser.add_argument("--cp-rank-max", type=int, default=20)
    parser.add_argument("--tucker-rank-min", type=int, default=1)
    parser.add_argument("--tucker-rank-max", type=int, default=20)
    parser.add_argument("--tucker-third-mode-rank", type=int, default=2)
    parser.add_argument(
        "--sigma-stab", type=float, default=None,
        help="Frozen scale param for s_k = exp(-d_stab/sigma_stab). Omit when fitting on a "
             "training split (a candidate value is logged at the end); pass the frozen value "
             "when scoring any split.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard to process (use with --num-shards)")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the query list into this many disjoint shards")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards.")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    process_file(
        args.s1, args.s2, args.s3, args.output,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
        cp_rank_range=range(args.cp_rank_min, args.cp_rank_max + 1),
        tucker_rank_range=range(args.tucker_rank_min, args.tucker_rank_max + 1),
        tucker_third_mode_rank=args.tucker_third_mode_rank,
        sigma_stab=args.sigma_stab, limit=args.limit,
        shard_index=args.shard_index, num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
