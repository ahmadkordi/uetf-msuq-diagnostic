#!/usr/bin/env python3
"""Clean claim-source ablation for the tensor scores (NO new NLI; uses the stored S1 matrices). Own variants, NOT the official MS-UQ construction.

Two single-factor removals of the claim source (S2), each compared with its official counterpart on the SAME test queries:
  CP:     official CP on 4 slices [S1agr, S2agr, S1dis, S2dis]  vs  CP on the 2 semantic slices [S1agr, S1dis]  (only the claim slices removed)
  Tucker: official Tucker on 2 slices [S1agr, S2agr], third-mode rank 2  vs  Tucker on the single slice S1agr (20x20x1, ranks [r, r, 1])
Scores: U = sum_r (1 - fit) with the same rank ranges and seed (20260903) as tensor_decompose.py (CP r=10..20, Tucker r=1..20).
Stored official scores (C.rawCP, C.rawTK) are the references; the first --check queries of the official construction are recomputed and compared with the stored values.
Paired query-bootstrap (n-boot, seed 1) of the AUROC difference (variant - official); both thresholds.
Usage (from /mnt/src): python tensor_clean_ablation.py --data-dir /mnt/output --cells <cells> --workers 8
Output: <data-dir>/experiment_0b/gate/tclean_<cell>.csv  (columns: cell,thr,pair,n,auroc_variant,auroc_official,diff,lo,hi)
"""
from __future__ import annotations
import argparse, csv
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from gate_eval import Cell
from baseline_auroc import official_auroc
from tensor_decompose import cp_rank_sweep, tucker_rank_sweep
from tensor_sem_only import load, official_scores, SEED


def clean_scores(args):
    A, D = args
    t2 = np.stack([np.asarray(A, float), np.asarray(D, float)], axis=-1)
    t1 = np.asarray(A, float)[:, :, None]
    cp = sum(r["uncertainty"] for r in cp_rank_sweep(t2, range(10, 21), seed=SEED))
    tk = sum(r["uncertainty"] for r in tucker_rank_sweep(t1, range(1, 21), third_mode_rank=1, seed=SEED))
    return cp, tk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--check", type=int, default=10); ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    O = a.data_dir; out = O / "experiment_0b" / "gate"
    for cell in a.cells:
        C = Cell(O, cell, "cal"); te = C.rows["test"]; ids = list(C.split["test"])
        s1 = load(O / f"stage_b_s1_semantic_{cell}.jsonl"); s2 = load(O / f"stage_b_s2_claim_{cell}.jsonl")
        with Pool(a.workers) as p:
            chk = p.map(official_scores, [(s1[ids[i]], s2[ids[i]]) for i in range(min(a.check, len(ids)))])
            res = p.map(clean_scores, [(s1[q]["matrix_agreement"], s1[q]["matrix_disagreement"]) for q in ids], chunksize=8)
        ocp, otk = C.rawCP[te], C.rawTK[te]
        print(f"[{cell}] reproducibility (first {len(chk)}): max|CP diff|={max(abs(c[0]-ocp[k]) for k, c in enumerate(chk)):.2e}, "
              f"max|Tucker diff|={max(abs(c[1]-otk[k]) for k, c in enumerate(chk)):.2e}", flush=True)
        vcp, vtk = np.array([r[0] for r in res]), np.array([r[1] for r in res])
        pairs = {"CPsem-CP4": (vcp, ocp), "TuckerS1-TuckerS1S2": (vtk, otk)}
        n = len(ids); rng = np.random.default_rng(1); idxs = [rng.integers(0, n, n) for _ in range(a.n_boot)]; rows = []
        for thr in (90, 70):
            acc = C.acc[thr][te]
            for name, (v, o) in pairs.items():
                av, ao = official_auroc(v, acc, C.N), official_auroc(o, acc, C.N)
                d = np.array([official_auroc(v[i], acc[i], C.N) - official_auroc(o[i], acc[i], C.N) for i in idxs])
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append({"cell": cell, "thr": thr, "pair": name, "n": n, "auroc_variant": av, "auroc_official": ao, "diff": av - ao, "lo": lo, "hi": hi})
        with open(out / f"tclean_{cell}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"[{cell}] done", flush=True)


if __name__ == "__main__":
    main()
