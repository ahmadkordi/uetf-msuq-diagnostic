#!/usr/bin/env python3
"""Aggregate bootstrap shards -> stability signal s_k (Signal B), sigma_stab fit on TRAIN only.

Inputs (per cell):
  * <shard-dir>/<cell>.shard<k>of<n>.jsonl   from run_bootstrap_sharded.py (raw replicates stored,
                                             run WITHOUT --sigma-stab)
  * <splits-dir>/<cell>.train_ids.txt        sigma_stab and the e-CDF are fit on these ids only
  * <trainfit-dir>/stage_g_trainfit_<cell>.jsonl   calibrate_ecdf.py output fit on TRAIN only
                                             (its "raw" dict supplies the train values of each e-CDF)

Two variants of d_stab are produced from the SAME stored replicates (no recomputation):
  raw : d_stab_k = MAD_b(U_k^(b))                 literal methodology v9 S11.2, raw estimator scale
  cal : d_stab_k = MAD_b( ecdf_k(U_k^(b)) )       replicates mapped through the train-fit e-CDF, i.e. the
                                                  same [0,1] scale as tilde U_k used by Signal A
Reason for two variants: raw estimators live on very different scales (U_CP reaches ~9, degree is in
[0,1]); one pooled sigma_stab on raw MADs would make s_k mostly a function of estimator scale. v9 does not
say which scale is meant, so both are written and the choice is reported, not hidden. The per-estimator
median d_stab (train) for both variants is stored in the summary so the scale effect is visible.

s_k = exp(-d_stab_k / sigma_stab),  sigma_stab = median of d_stab pooled over all train queries and all
estimators (same rule as bootstrap_stability.py's logged candidate), one sigma per cell and variant.

Failed replicates (None, singular ALS solve) are excluded from the MAD; an (query, estimator) pair with
fewer than MIN_VALID=3 valid replicates has no d_stab. Such pairs are IMPUTED with the cell's train-median
s_k of that estimator, flagged per record ("imputed") and counted in the summary.

Everything here is gold-free: no correctness labels are read.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from calibrate_ecdf import apply_ecdf, fit_ecdf

LOGGER = logging.getLogger("uetf.aggregate_bootstrap")
MIN_VALID = 3
CELLS = [
    f"{ds}_{m}"
    for ds in ("coqa", "hotpotqa", "nqopen", "truthfulqa")
    for m in ("Llama2_13b", "llama3.1_8b", "Q2.5_14b")
]


def mad(values: list[float]) -> float:
    med = statistics.median(values)
    return float(statistics.median(abs(v - med) for v in values))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_shards(shard_dir: Path, cell: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pat = re.compile(rf"{re.escape(cell)}\.shard(\d+)of(\d+)\.jsonl")
    found: dict[int, Path] = {}
    totals: set[int] = set()
    for path in shard_dir.iterdir():
        m = pat.fullmatch(path.name)
        if m:
            found[int(m.group(1))] = path
            totals.add(int(m.group(2)))
    info: dict[str, Any] = {"shards_found": len(found)}
    if len(totals) != 1:
        raise ValueError(f"{cell}: inconsistent or missing shard files (num_shards seen: {sorted(totals)})")
    n_shards = totals.pop()
    info["shards_expected"] = n_shards
    info["missing_shards"] = sorted(set(range(n_shards)) - set(found))
    records: dict[str, dict[str, Any]] = {}
    for path in found.values():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = str(rec["id"])
                if qid in records:
                    raise ValueError(f"{cell}: duplicate id {qid} across shards")
                records[qid] = rec
    return records, info


def valid_values(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def process_cell(
    cell: str, shard_dir: Path, splits_dir: Path, trainfit_dir: Path, out_dir: Path, allow_incomplete: bool,
) -> dict[str, Any]:
    records, info = load_shards(shard_dir, cell)
    train_ids = read_ids(splits_dir / f"{cell}.train_ids.txt")
    all_split_ids = set(train_ids) | set(read_ids(splits_dir / f"{cell}.val_ids.txt")) | set(
        read_ids(splits_dir / f"{cell}.test_ids.txt"))
    missing_ids = sorted(all_split_ids - set(records))
    extra_ids = sorted(set(records) - all_split_ids)
    info.update({"n_records": len(records), "n_split_ids": len(all_split_ids),
                 "n_missing_ids": len(missing_ids), "n_extra_ids": len(extra_ids)})
    if (info["missing_shards"] or missing_ids) and not allow_incomplete:
        raise ValueError(f"{cell}: incomplete ({len(info['missing_shards'])} shards, {len(missing_ids)} ids missing). "
                         "Finish the sharded run or pass --allow-incomplete.")
    train_ids = [i for i in train_ids if i in records]
    if not train_ids:
        raise ValueError(f"{cell}: no train ids present in shards.")

    # e-CDF sorted train samples, from the train-only calibration file ("raw" of train queries)
    raw_by_id: dict[str, dict[str, float]] = {}
    train_set = set(train_ids)
    with (trainfit_dir / f"stage_g_trainfit_{cell}.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec["id"]) in train_set:
                raw_by_id[str(rec["id"])] = rec["raw"]
            cal_meta = rec["calibration"]
    if cal_meta.get("same_set_smoke_test"):
        raise ValueError(f"{cell}: trainfit file is a same-set smoke test, not train-only. Rebuild it with --train-ids.")
    if set(raw_by_id) != train_set:
        raise ValueError(f"{cell}: trainfit file lacks {len(train_set - set(raw_by_id))} train ids.")
    names = sorted(next(iter(records.values()))["replicates"])
    sorted_train = {n: fit_ecdf(np.asarray([raw_by_id[i][n] for i in train_ids], dtype=float)) for n in names}

    # per-query, per-estimator d_stab for both variants; failure accounting; consistency check
    fail_counts = {n: 0 for n in names}
    total_replicates = 0
    d_raw: dict[str, dict[str, float | None]] = {}
    d_cal: dict[str, dict[str, float | None]] = {}
    mismatches = 0
    for qid, rec in records.items():
        d_raw[qid], d_cal[qid] = {}, {}
        for n in names:
            reps = rec["replicates"][n]
            if len(reps) != rec["n_bootstrap"]:
                raise ValueError(f"{cell} id={qid} {n}: {len(reps)} replicates, expected {rec['n_bootstrap']}")
            ok = valid_values(reps)
            total_replicates += len(reps)
            fail_counts[n] += len(reps) - len(ok)
            if len(ok) >= MIN_VALID:
                d_raw[qid][n] = mad(ok)
                d_cal[qid][n] = mad([apply_ecdf(sorted_train[n], v) for v in ok])
            else:
                d_raw[qid][n] = d_cal[qid][n] = None
            stored = rec.get("d_stab", {}).get(n, "absent")
            if stored != "absent" and (stored is None) != (d_raw[qid][n] is None):
                mismatches += 1
            elif stored not in ("absent", None) and d_raw[qid][n] is not None and abs(stored - d_raw[qid][n]) > 1e-9:
                mismatches += 1

    summary: dict[str, Any] = {"shards": info, "total_replicates": total_replicates,
                               "failed_replicates": {n: c for n, c in fail_counts.items() if c},
                               "n_failed_replicates_total": sum(fail_counts.values()),
                               "d_stab_mismatch_vs_stored": mismatches, "variants": {}}
    n_pairs_none = sum(1 for q in d_raw for n in names if d_raw[q][n] is None)
    summary["pairs_without_d_stab"] = n_pairs_none

    for variant, d in (("raw", d_raw), ("cal", d_cal)):
        pooled = [d[q][n] for q in train_ids for n in names if d[q][n] is not None]
        sigma = float(statistics.median(pooled))
        if not sigma > 0:
            raise ValueError(f"{cell}/{variant}: sigma_stab={sigma} (median pooled d_stab is 0). "
                             "Refusing to guess a replacement -- inspect the d_stab distribution.")
        per_est_median = {n: float(statistics.median([d[q][n] for q in train_ids if d[q][n] is not None]))
                          for n in names if any(d[q][n] is not None for q in train_ids)}
        s_train_median = {n: float(statistics.median([math.exp(-d[q][n] / sigma) for q in train_ids
                                                      if d[q][n] is not None])) for n in per_est_median}
        out_path = out_dir / f"bootstrap_sk_{variant}_{cell}.jsonl"
        n_imputed = 0
        with out_path.open("w", encoding="utf-8") as dst:
            for qid in records:
                s_k, imputed = {}, []
                for n in names:
                    if d[qid][n] is None:
                        s_k[n] = s_train_median[n]
                        imputed.append(n)
                    else:
                        s_k[n] = float(math.exp(-d[qid][n] / sigma))
                n_imputed += len(imputed)
                dst.write(json.dumps({"id": qid, "variant": variant, "sigma_stab": sigma, "s_k": s_k,
                                      "d_stab": d[qid], "imputed": imputed}, ensure_ascii=False) + "\n")
        summary["variants"][variant] = {"sigma_stab": sigma, "n_pairs_imputed": n_imputed,
                                        "train_median_d_stab_per_estimator": per_est_median,
                                        "output": str(out_path)}
        LOGGER.info("%s/%s: sigma_stab=%.6f imputed_pairs=%d -> %s", cell, variant, sigma, n_imputed, out_path)
    LOGGER.info("%s: %d records, failed replicates=%d/%d, pairs without d_stab=%d, d_stab mismatches=%d",
                cell, len(records), summary["n_failed_replicates_total"], total_replicates, n_pairs_none, mismatches)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-dir", type=Path, default=Path("/mnt/output/experiment_0b/bootstrap_shards"))
    parser.add_argument("--splits-dir", type=Path, default=Path("/mnt/output/experiment_0a/splits"))
    parser.add_argument("--trainfit-dir", type=Path, default=Path("/mnt/output/experiment_0b"))
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/output/experiment_0b"))
    parser.add_argument("--summary-out", type=Path, default=Path("/mnt/output/experiment_0b/bootstrap_aggregate_summary.json"))
    parser.add_argument("--cells", nargs="+", default=CELLS)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, failures = {}, []
    for cell in args.cells:
        try:
            summaries[cell] = process_cell(cell, args.shard_dir, args.splits_dir, args.trainfit_dir,
                                           args.out_dir, args.allow_incomplete)
        except (ValueError, FileNotFoundError) as exc:
            LOGGER.error("%s: %s", cell, exc)
            failures.append(cell)
    args.summary_out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    LOGGER.info("summary -> %s", args.summary_out)
    if failures:
        LOGGER.error("cells not processed: %s", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
