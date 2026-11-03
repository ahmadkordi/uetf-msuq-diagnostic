#!/usr/bin/env python3
"""Step 1 of the 'tensor without symmetrisation' check: write the first --n TEST queries of one cell (in test-split order) from the
Stage-A responses file and the Stage-B claims file into two small JSONL inputs for semantic_nli.py / claim_nli.py --no-symmetrize.
Usage (from /mnt/src): python nosym_subset.py --data-dir /mnt/output --cell coqa_llama3.1_8b --stage-a <path> --claims <path> --n 1000 --out-dir /mnt/output/nosym
"""
import argparse, json
from pathlib import Path
from gate_eval import Cell

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cell", required=True)
ap.add_argument("--stage-a", type=Path, required=True); ap.add_argument("--claims", type=Path, required=True)
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out-dir", type=Path, required=True); a = ap.parse_args()
C = Cell(a.data_dir, a.cell, "cal"); ids = [str(i) for i in C.split["test"]][: a.n]; keep = set(ids)
a.out_dir.mkdir(parents=True, exist_ok=True)
for src, name in ((a.stage_a, "stage_a_sub.jsonl"), (a.claims, "claims_sub.jsonl")):
    rows = {}
    with open(src, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                if str(r["id"]) in keep: rows[str(r["id"])] = ln if ln.endswith("\n") else ln + "\n"
    miss = [i for i in ids if i not in rows]
    with open(a.out_dir / name, "w", encoding="utf-8") as fh:
        for i in ids:
            if i in rows: fh.write(rows[i])
    print(f"{name}: {len(rows)} written, {len(miss)} missing")
json.dump(ids, open(a.out_dir / "ids.json", "w"))
