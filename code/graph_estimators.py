#!/usr/bin/env python3
"""Stage F: Graph-based candidate estimator pool for U-ETF.

Computes EigV, Degree, and Eccentricity uncertainty estimators on the
Laplacian of each (S1/S2 x Agreement/Disagreement) matrix -- 12 estimators
total -- plus surfaces the scalar U_p from S^(3) for convenience.

IMPORTANT audit-honesty note (unlike semantic_nli.py / claim_nli.py /
tensor_decompose.py, which were checked line-by-line against
github.com/tiejin98/MSUQ):
    That repository does NOT contain an implementation of EigV, Degree, or
    Eccentricity -- only the normalized-Laplacian helper
    (transfer_matrix.py: get_D_mat / get_L_mat) is present. These three
    estimators originate from Lin, Trivedi & Sun, "Generating with
    Confidence" (TMLR 2023), which the MS-UQ paper cites as its baseline
    comparison, but whose code is not reproduced in this repo. The formulas
    below are the standard, widely-cited definitions from that paper --
    EigV and Degree are unambiguous and implemented with high confidence;
    Eccentricity involves a truncation/embedding choice that varies
    slightly across re-implementations in the literature, and is NOT
    verified against an official codebase here. Before reporting
    Eccentricity numbers as a faithful baseline reproduction, cross-check
    against Lin et al.'s own released code.

Definitions (W = similarity matrix, symmetric, diag(W) = 1; D = diag(row
sums of W); L = D^-1/2 (D - W) D^-1/2, the normalized graph Laplacian,
eigenvalues 0 <= lambda_1 <= ... <= lambda_N):

    U_Degree = 1 - mean(W)                          (= trace(N*I - D) / N^2)
    U_EigV   = sum_k max(0, 1 - lambda_k)
    U_Ecc    : spectral embedding using eigenvectors with lambda_k < 1,
               each column scaled by sqrt(max(0, 1 - lambda_k)); per-response
               eccentricity = distance of its embedding row from the
               embedding centroid; U_Ecc = L2 norm of the per-response
               eccentricity vector. (This truncation choice -- reuse the
               same "informative" eigenvectors that EigV counts -- is a
               documented, reasonable default, not an audited one.)

This module is deliberately independent of correctness labels, gold
answers, and the U-ETF reliability/gating layer.

Input:  Stage-B S^(1) file (semantic_nli.py) and S^(2) file (claim_nli.py),
        plus optionally the S^(3) file (self_eval.py) for U_p.
Output: JSONL with all 12 (+1) per-query estimator values, persisted
        individually (not just an ensembled score).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.graph_estimators")


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


def normalized_laplacian(w: np.ndarray) -> np.ndarray:
    """L = D^-1/2 (D - W) D^-1/2, matching transfer_matrix.py's get_L_mat."""
    d = w.sum(axis=1)
    if np.any(d <= 0):
        raise ValueError("Degree matrix has non-positive entries; check diag(W)=1 was enforced upstream.")
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    laplacian = d_inv_sqrt @ (np.diag(d) - w) @ d_inv_sqrt
    # Symmetrize away floating-point noise (L is symmetric by construction).
    return 0.5 * (laplacian + laplacian.T)


def eigv_uncertainty(eigenvalues: np.ndarray) -> float:
    return float(np.sum(np.maximum(0.0, 1.0 - eigenvalues)))


def degree_uncertainty(w: np.ndarray) -> float:
    return float(1.0 - w.mean())


def eccentricity_uncertainty(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> float:
    weights = np.maximum(0.0, 1.0 - eigenvalues)
    keep = weights > 1e-12
    if not np.any(keep):
        # No informative eigenvectors (fully agreed / fully disagreed matrix):
        # every response embeds at the same point -> zero eccentricity.
        return 0.0
    v = eigenvectors[:, keep] * np.sqrt(weights[keep])  # (N, c)
    centroid = v.mean(axis=0, keepdims=True)
    per_response = np.linalg.norm(v - centroid, axis=1)  # (N,)
    return float(np.linalg.norm(per_response))


def estimators_for_matrix(w: np.ndarray) -> dict[str, float]:
    laplacian = normalized_laplacian(w)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)  # ascending order
    return {
        "eigv": eigv_uncertainty(eigenvalues),
        "degree": degree_uncertainty(w),
        "eccentricity": eccentricity_uncertainty(eigenvalues, eigenvectors),
    }


def compute_all_estimators(s1: dict[str, Any], s2: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source_name, record in (("s1_semantic", s1), ("s2_claim", s2)):
        for variant in ("agreement", "disagreement"):
            w = np.asarray(record[f"matrix_{variant}"], dtype=np.float64)
            scores = estimators_for_matrix(w)
            for name, value in scores.items():
                out[f"{source_name}.{variant}.{name}"] = value
    return out


def process_file(
    s1_path: Path,
    s2_path: Path,
    output_path: Path,
    *,
    s3_path: Path | None,
    limit: int | None,
) -> None:
    s1_records = load_keyed(s1_path)
    s2_records = load_keyed(s2_path)
    s3_records = load_keyed(s3_path) if s3_path is not None else {}

    ids = [i for i in s1_records if i in s2_records]
    missing = set(s1_records) ^ set(s2_records)
    if missing:
        LOGGER.warning("IDs present in only one of S1/S2, skipped: %s", sorted(missing))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for query_id in ids:
            if limit is not None and processed >= limit:
                break
            s1, s2 = s1_records[query_id], s2_records[query_id]
            question = s1.get("question", s2.get("question", ""))

            estimators = compute_all_estimators(s1, s2)

            if query_id in s3_records:
                estimators["s3_self_eval.U_p"] = s3_records[query_id]["p_true_scalar_uncertainty"]

            result = {
                "id": query_id,
                "question": question,
                "estimators": estimators,
                "meta": {
                    "laplacian": "D^-1/2 (D - W) D^-1/2",
                    "eccentricity_audited": False,
                    "eccentricity_source": (
                        "Lin, Trivedi & Sun (2023) 'Generating with Confidence' -- "
                        "not present in github.com/tiejin98/MSUQ; verify against "
                        "their own code before treating as baseline-faithful."
                    ),
                },
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            LOGGER.info("id=%s estimators=%d", query_id, len(estimators))

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", required=True, type=Path, help="semantic_nli.py output")
    parser.add_argument("--s2", required=True, type=Path, help="claim_nli.py output")
    parser.add_argument("--s3", type=Path, default=None, help="self_eval.py output (optional, adds U_p)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    process_file(args.s1, args.s2, args.output, s3_path=args.s3, limit=args.limit)


if __name__ == "__main__":
    main()
