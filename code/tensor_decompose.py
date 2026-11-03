#!/usr/bin/env python3
"""Stage C/D: Tensor construction + CP/Tucker decomposition for U-ETF.

Deliberately implemented in pure Python (tensorly), not MATLAB, so the
whole pipeline stays on one platform/toolchain -- a project requirement,
not just a convenience. Install with: pip install tensorly

Phase 0 audit note (github.com/tiejin98/MSUQ, CP_decom.m / Tucker_decom.m):
    The two decompositions are NOT symmetric in the official code:
      - CP tensor:     4 slices -> [S1_agree, S2_agree, S1_disagree, S2_disagree]
                        rank swept over r = 10..20 (near-full-rank sweep, N=20)
      - Tucker tensor: 2 slices -> [S1_agree, S2_agree]  (agreement only)
                        rank swept over r = 1..20 for modes 1-2,
                        mode-3 rank fixed at 2 (= full rank on that mode,
                        since there are only 2 slices)
    We reproduce that asymmetry exactly rather than "cleaning it up" into a
    single shared tensor, since that would silently change what is being
    measured relative to the official baseline.

    Per-rank uncertainty is 1 - fit, where
        fit = 1 - ||X - X_hat||_F / ||X||_F
    and the final U_CP / U_Tucker score is the SUM of (1 - fit) over the
    swept rank range (a "rank-ensemble" signal, not a single best-fit
    number) -- matches get_matlab_res.py / inter_unq_ensemble.py.

Self-evaluation (S^3) does NOT enter either tensor in the official
baseline; it stays a separate source (see self_eval.py). Fusing it into
the tensor is a v9-specific extension, not implemented here to keep this
module a faithful baseline reproduction -- extend deliberately, in a
clearly-labelled separate function, if/when you want that ablation.

This module is deliberately independent of correctness labels, gold
answers, and the U-ETF reliability/gating layer -- it only ever sees
source matrices.

Input:  Stage-B S^(1) file (semantic_nli.py output) and Stage-B S^(2) file
        (claim_nli.py output), matched by query id.
Output: JSONL with, per query, every per-rank fit/uncertainty value plus
        the summed U_CP and U_Tucker scores.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.tensor_decompose")


def load_source_file(path: Path) -> dict[str, dict[str, Any]]:
    """Load a Stage-B (S^1 or S^2) JSONL, keyed by query id."""
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


def _matrix(record: dict[str, Any], key: str) -> np.ndarray:
    if key not in record:
        raise ValueError(f"id={record.get('id')}: missing '{key}'.")
    return np.asarray(record[key], dtype=np.float64)


def build_cp_tensor(s1: dict[str, Any], s2: dict[str, Any]) -> np.ndarray:
    """4-slice tensor: [S1_agree, S2_agree, S1_disagree, S2_disagree]."""
    slices = [
        _matrix(s1, "matrix_agreement"),
        _matrix(s2, "matrix_agreement"),
        _matrix(s1, "matrix_disagreement"),
        _matrix(s2, "matrix_disagreement"),
    ]
    n = slices[0].shape[0]
    for m in slices:
        if m.shape != (n, n):
            raise ValueError(f"Inconsistent matrix shapes: expected {(n, n)}, got {m.shape}")
    return np.stack(slices, axis=-1)  # (N, N, 4)


def build_tucker_tensor(s1: dict[str, Any], s2: dict[str, Any]) -> np.ndarray:
    """2-slice tensor: [S1_agree, S2_agree] (agreement only, per baseline)."""
    slices = [
        _matrix(s1, "matrix_agreement"),
        _matrix(s2, "matrix_agreement"),
    ]
    n = slices[0].shape[0]
    for m in slices:
        if m.shape != (n, n):
            raise ValueError(f"Inconsistent matrix shapes: expected {(n, n)}, got {m.shape}")
    return np.stack(slices, axis=-1)  # (N, N, 2)


def _fit(x: np.ndarray, x_hat: np.ndarray) -> float:
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return 1.0 if np.linalg.norm(x_hat) == 0 else 0.0
    return float(1.0 - np.linalg.norm(x - x_hat) / norm_x)


def cp_rank_sweep(
    tensor: np.ndarray,
    rank_range: range,
    *,
    n_iter_max: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    import tensorly as tl
    from tensorly.decomposition import parafac

    tl.set_backend("numpy")
    x = tl.tensor(tensor)
    results = []
    for r in rank_range:
        weights, factors = parafac(
            x, rank=r, n_iter_max=n_iter_max, init="random", random_state=seed, normalize_factors=False
        )
        x_hat = tl.cp_to_tensor((weights, factors))
        fit = _fit(tensor, np.asarray(x_hat))
        results.append({"rank": r, "fit": fit, "uncertainty": 1.0 - fit})
    return results


def tucker_rank_sweep(
    tensor: np.ndarray,
    rank_range: range,
    *,
    third_mode_rank: int,
    n_iter_max: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    import tensorly as tl
    from tensorly.decomposition import tucker

    tl.set_backend("numpy")
    x = tl.tensor(tensor)
    third = min(third_mode_rank, tensor.shape[2])
    results = []
    for r in rank_range:
        rank = [min(r, tensor.shape[0]), min(r, tensor.shape[1]), third]
        core, factors = tucker(x, rank=rank, n_iter_max=n_iter_max, init="random", random_state=seed)
        x_hat = tl.tucker_to_tensor((core, factors))
        fit = _fit(tensor, np.asarray(x_hat))
        results.append({"rank": r, "fit": fit, "uncertainty": 1.0 - fit})
    return results


def process_file(
    s1_path: Path,
    s2_path: Path,
    output_path: Path,
    *,
    cp_rank_range: range,
    tucker_rank_range: range,
    tucker_third_mode_rank: int,
    seed: int,
    limit: int | None,
) -> None:
    s1_records = load_source_file(s1_path)
    s2_records = load_source_file(s2_path)
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

            cp_tensor = build_cp_tensor(s1, s2)
            tucker_tensor = build_tucker_tensor(s1, s2)

            cp_results = cp_rank_sweep(cp_tensor, cp_rank_range, seed=seed)
            tucker_results = tucker_rank_sweep(
                tucker_tensor, tucker_rank_range, third_mode_rank=tucker_third_mode_rank, seed=seed
            )

            u_cp = float(sum(r["uncertainty"] for r in cp_results))
            u_tucker = float(sum(r["uncertainty"] for r in tucker_results))

            result = {
                "id": query_id,
                "question": question,
                "tensor": {
                    "cp_slices": ["S1_agreement", "S2_agreement", "S1_disagreement", "S2_disagreement"],
                    "cp_shape": list(cp_tensor.shape),
                    "cp_rank_range": [cp_rank_range.start, cp_rank_range.stop - 1],
                    "tucker_slices": ["S1_agreement", "S2_agreement"],
                    "tucker_shape": list(tucker_tensor.shape),
                    "tucker_rank_range": [tucker_rank_range.start, tucker_rank_range.stop - 1],
                    "tucker_third_mode_rank": tucker_third_mode_rank,
                    "seed": seed,
                },
                "cp_per_rank": cp_results,
                "tucker_per_rank": tucker_results,
                "U_CP": u_cp,
                "U_Tucker": u_tucker,
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            LOGGER.info("id=%s U_CP=%.4f U_Tucker=%.4f", query_id, u_cp, u_tucker)

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def self_test() -> None:
    """Sanity check: a synthetic low-rank tensor should reconstruct with
    fit -> 1 (uncertainty -> 0) as rank grows. Run this after installing
    tensorly to confirm the library/backend behave as expected in your
    environment -- this sandbox has no network access to verify it here.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(0)
    true_rank = 3
    a = rng.standard_normal((20, true_rank))
    b = rng.standard_normal((20, true_rank))
    c = rng.standard_normal((4, true_rank))
    tensor = np.einsum("ir,jr,kr->ijk", a, b, c)

    for r in (1, 2, 3, 5, 10):
        results = cp_rank_sweep(tensor, range(r, r + 1), n_iter_max=200, seed=0)
        print(f"CP rank={r}: fit={results[0]['fit']:.4f} uncertainty={results[0]['uncertainty']:.4f}")

    tucker_tensor = tensor[:, :, :2]
    for r in (1, 2, 5, 10, 20):
        results = tucker_rank_sweep(tucker_tensor, range(r, r + 1), third_mode_rank=2, n_iter_max=200, seed=0)
        print(f"Tucker rank={r}: fit={results[0]['fit']:.4f} uncertainty={results[0]['uncertainty']:.4f}")
    print("Expect fit -> ~1.0 (uncertainty -> ~0.0) once rank >= true_rank=3 for CP.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, help="Stage-B S^(1) semantic_nli.py output")
    parser.add_argument("--s2", type=Path, help="Stage-B S^(2) claim_nli.py output")
    parser.add_argument("--output", type=Path, help="Output JSONL")
    parser.add_argument("--cp-rank-min", type=int, default=10)
    parser.add_argument("--cp-rank-max", type=int, default=20)
    parser.add_argument("--tucker-rank-min", type=int, default=1)
    parser.add_argument("--tucker-rank-max", type=int, default=20)
    parser.add_argument("--tucker-third-mode-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic sanity check and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    if args.self_test:
        self_test()
        return

    if not (args.s1 and args.s2 and args.output):
        raise SystemExit("--s1, --s2 and --output are required unless --self-test is given.")

    process_file(
        args.s1,
        args.s2,
        args.output,
        cp_rank_range=range(args.cp_rank_min, args.cp_rank_max + 1),
        tucker_rank_range=range(args.tucker_rank_min, args.tucker_rank_max + 1),
        tucker_third_mode_rank=args.tucker_third_mode_rank,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
