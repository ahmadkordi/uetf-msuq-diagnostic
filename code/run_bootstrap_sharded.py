#!/usr/bin/env python3
"""Run bootstrap_stability.py over all 12 cells, split into shards, P at a time.
No --sigma-stab is passed: the expensive bootstrap replicates are computed and stored
once; s_k is derived afterwards from the stored replicates (sigma is fit on the train
split later). Resumable: a finished shard file is never recomputed; a shard is written to
"<name>.part" and renamed only on success.

Usage (inside tmux, venv active):
    python run_bootstrap_sharded.py --parallel 60
    python run_bootstrap_sharded.py --dry-run          # print the job list only
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOGGER = logging.getLogger("uetf.run_bootstrap_sharded")

CELLS = [
    f"{ds}_{m}"
    for ds in ("coqa", "hotpotqa", "nqopen", "truthfulqa")
    for m in ("Llama2_13b", "llama3.1_8b", "Q2.5_14b")
]


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_jobs(data_dir: Path, out_dir: Path, shard_size: int) -> list[tuple[str, int, int, Path]]:
    jobs = []
    for cell in CELLS:
        for stem in ("stage_b_s1_semantic", "stage_b_s2_claim", "stage_b_s3_self_eval"):
            path = data_dir / f"{stem}_{cell}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing input: {path}")
        n = count_nonempty_lines(data_dir / f"stage_b_s1_semantic_{cell}.jsonl")
        num_shards = max(1, -(-n // shard_size))
        for k in range(num_shards):
            jobs.append((cell, k, num_shards, out_dir / f"{cell}.shard{k}of{num_shards}.jsonl"))
    return jobs


def run_job(job: tuple[str, int, int, Path], data_dir: Path, src_dir: Path, env: dict[str, str]) -> tuple[str, str]:
    cell, k, num_shards, out = job
    if out.exists() and out.stat().st_size > 0:
        return str(out), "skip"
    part = out.with_name(out.name + ".part")
    log = out.with_suffix(".log")
    cmd = [
        sys.executable, str(src_dir / "bootstrap_stability.py"),
        "--s1", str(data_dir / f"stage_b_s1_semantic_{cell}.jsonl"),
        "--s2", str(data_dir / f"stage_b_s2_claim_{cell}.jsonl"),
        "--s3", str(data_dir / f"stage_b_s3_self_eval_{cell}.jsonl"),
        "--output", str(part),
        "--shard-index", str(k), "--num-shards", str(num_shards),
    ]
    with log.open("w") as log_handle:
        rc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        return str(out), f"FAILED rc={rc} (see {log})"
    part.rename(out)
    return str(out), "done"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/output"))
    parser.add_argument("--src-dir", type=Path, default=Path("/mnt/src"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data-dir>/experiment_0b/bootstrap_shards")
    parser.add_argument("--parallel", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=400, help="queries per shard")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    out_dir = args.out_dir or (args.data_dir / "experiment_0b" / "bootstrap_shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.data_dir, out_dir, args.shard_size)
    LOGGER.info("jobs: %d  parallel: %d  shard_size: %d  out: %s", len(jobs), args.parallel, args.shard_size, out_dir)
    if args.dry_run:
        for cell, k, ns, out in jobs:
            print(cell, k, ns, out.name)
        return

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"

    start = time.time()
    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_job, job, args.data_dir, args.src_dir, env) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            name, status = fut.result()
            if status == "done":
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                LOGGER.error("%s %s", name, status)
            if i % 5 == 0 or i == len(jobs):
                LOGGER.info("progress %d/%d  done=%d skipped=%d failed=%d  elapsed=%.1f min",
                            i, len(jobs), done, skipped, failed, (time.time() - start) / 60)
    if failed:
        LOGGER.error("%d shard(s) failed -- rerun the same command to retry only those.", failed)
        sys.exit(1)
    LOGGER.info("ALL SHARDS DONE")


if __name__ == "__main__":
    main()
