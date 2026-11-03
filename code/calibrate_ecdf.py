#!/usr/bin/env python3
"""Stage G: Empirical-CDF (e-CDF) calibration of the Stage F estimator pool.

For each raw estimator, fits an empirical CDF on a TRAIN set of queries and
applies it to a (possibly different) EVAL set, mapping raw estimator values
onto a comparable [0, 1] scale: calibrated(x) = P_train(raw_estimator <= x).

This module is deliberately independent of correctness labels and gold
answers -- calibration here is purely distributional (it only looks at the
estimator's own value distribution across queries), never at whether a
response was actually correct.

*** SMALL-N WARNING - read before trusting any number this script prints ***
With only 25 pilot queries, an e-CDF fit on (a subset of) them is not a
statistically meaningful calibration -- it is provided so you can exercise
the full Stage A-G pipeline end to end on the data you have now. Do not
report calibrated values from this run in the paper; re-run against the
real CoQA/NQ-Open/HotpotQA/TruthfulQA benchmark splits once available. The
script deliberately keeps train/eval as separate CLI args (not silently the
same file) so a real split is a natural next step, not a rewrite.

Orientation note: every raw estimator here is already defined so that
LARGER = MORE UNCERTAIN (U_Degree, U_EigV, U_Ecc, U_CP, U_Tucker, U_p all
increase with disagreement/entropy). No sign-flip ("orientation check") is
applied. If you add an estimator where smaller = more uncertain, flip its
sign before calibration, not after -- flag it explicitly rather than
silently negating inside this script.

Input:  one or more Stage-F files (graph_estimators.py output), and
        optionally the Stage-C/D tensor file (tensor_decompose.py output)
        to fold U_CP/U_Tucker into the same calibrated pool.
Output: JSONL with, per eval query, every estimator's raw AND calibrated
        value.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.calibrate_ecdf")


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


def merged_estimators(
    query_id: str,
    stage_f: dict[str, dict[str, Any]],
    tensor: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    """Combine Stage-F graph/self-eval estimators with Stage-C/D tensor
    estimators (U_CP, U_Tucker) into one flat dict, if tensor data is given.
    """
    out: dict[str, float] = dict(stage_f[query_id]["estimators"])
    if tensor is not None and query_id in tensor:
        out["tensor.U_CP"] = tensor[query_id]["U_CP"]
        out["tensor.U_Tucker"] = tensor[query_id]["U_Tucker"]
    return out


def fit_ecdf(values: np.ndarray) -> np.ndarray:
    """Return the sorted training values needed to evaluate an e-CDF via
    searchsorted. Ties are handled by 'right' search (>= counts as covered).
    """
    return np.sort(values)


def apply_ecdf(sorted_train: np.ndarray, x: float) -> float:
    if len(sorted_train) == 0:
        raise ValueError("Empty training set for e-CDF.")
    rank = np.searchsorted(sorted_train, x, side="right")
    return float(rank) / float(len(sorted_train))


def calibrate(
    train_ids: list[str],
    eval_ids: list[str],
    stage_f: dict[str, dict[str, Any]],
    tensor: dict[str, dict[str, Any]] | None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Returns {query_id: {estimator_name: calibrated_value}} for eval_ids,
    plus the sorted list of estimator names used (for reporting)."""
    train_raw: dict[str, list[float]] = {}
    for qid in train_ids:
        for name, value in merged_estimators(qid, stage_f, tensor).items():
            train_raw.setdefault(name, []).append(value)

    estimator_names = sorted(train_raw)
    sorted_train = {name: fit_ecdf(np.asarray(train_raw[name])) for name in estimator_names}

    results: dict[str, dict[str, float]] = {}
    for qid in eval_ids:
        raw = merged_estimators(qid, stage_f, tensor)
        results[qid] = {
            name: apply_ecdf(sorted_train[name], raw[name])
            for name in estimator_names
            if name in raw
        }
    return results, estimator_names


def process(
    stage_f_paths: list[Path],
    tensor_path: Path | None,
    train_ids_path: Path | None,
    eval_ids_path: Path | None,
    output_path: Path,
) -> None:
    stage_f: dict[str, dict[str, Any]] = {}
    for path in stage_f_paths:
        stage_f.update(load_keyed(path))
    tensor = load_keyed(tensor_path) if tensor_path is not None else None

    all_ids = sorted(stage_f)
    if train_ids_path is not None:
        train_ids = [line.strip() for line in train_ids_path.read_text().splitlines() if line.strip()]
    else:
        LOGGER.warning(
            "No --train-ids given: fitting the e-CDF on the SAME %d queries being "
            "calibrated. This is only valid for pipeline smoke-testing -- see the "
            "SMALL-N WARNING in this script's docstring.",
            len(all_ids),
        )
        train_ids = all_ids
    eval_ids = (
        [line.strip() for line in eval_ids_path.read_text().splitlines() if line.strip()]
        if eval_ids_path is not None
        else all_ids
    )

    missing_train = [i for i in train_ids if i not in stage_f]
    missing_eval = [i for i in eval_ids if i not in stage_f]
    if missing_train or missing_eval:
        raise ValueError(f"IDs missing from Stage-F data: train={missing_train} eval={missing_eval}")

    calibrated, estimator_names = calibrate(train_ids, eval_ids, stage_f, tensor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        for qid in eval_ids:
            raw = merged_estimators(qid, stage_f, tensor)
            result = {
                "id": qid,
                "question": stage_f[qid].get("question", ""),
                "calibration": {
                    "train_n": len(train_ids),
                    "eval_n": len(eval_ids),
                    "same_set_smoke_test": train_ids_path is None,
                    "estimator_names": estimator_names,
                },
                "raw": raw,
                "calibrated": calibrated[qid],
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")

    LOGGER.info("Calibrated %d queries against %d training queries. Output=%s", len(eval_ids), len(train_ids), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-f", required=True, type=Path, nargs="+", help="One or more graph_estimators.py outputs")
    parser.add_argument("--tensor", type=Path, default=None, help="Optional tensor_decompose.py output to fold in U_CP/U_Tucker")
    parser.add_argument("--train-ids", type=Path, default=None, help="Text file, one query id per line, to fit the e-CDF on")
    parser.add_argument("--eval-ids", type=Path, default=None, help="Text file, one query id per line, to calibrate (default: all)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    process(args.stage_f, args.tensor, args.train_ids, args.eval_ids, args.output)


if __name__ == "__main__":
    main()
