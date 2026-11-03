#!/usr/bin/env python3
"""Semantic-only tensor ablation (NO new NLI; uses the stored S1 matrices). Own variant, NOT the official MS-UQ construction.

Tensor: 2 slices [S1_agreement, S1_disagreement] (20 x 20 x 2); no claim source.
  U_CPsem     = sum over r=10..20 of (1 - fit), CP (parafac, random init, n_iter_max=200, seed = stage C seed 20260903)   [as tensor_decompose.cp_rank_sweep]
  U_Tsem      = sum over r=1..20 of (1 - fit), Tucker rank [r, r, 2]                                                      [as tucker_rank_sweep]
  U_Sumsem    = U_CPsem + U_Tsem  (raw sums, like B3-Sum)
Reference on the SAME queries: stored B3-Sum (raw CP + raw Tucker of stage C), CP-only, Tucker-only.
A reproducibility check recomputes the OFFICIAL (4-slice CP, 2-slice agreement Tucker) scores for the first --check queries and prints the max abs difference to the stored values.
Evaluates on a fixed random subset of the TEST split (--max-n per cell, default all) with the official AUROC and a paired bootstrap vs B3-Sum.
Usage (from /mnt/src): python tensor_sem_only.py --data-dir /mnt/output --cells <cells> --max-n 600 --workers 8
Output: <data-dir>/experiment_0b/gate/tsem_<cell>.csv
"""
from __future__ import annotations
import argparse, csv, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from gate_eval import Cell
from baseline_auroc import official_auroc
from tensor_decompose import cp_rank_sweep, tucker_rank_sweep, build_cp_tensor, build_tucker_tensor

SEED = 20260903


def sem_scores(args):
    A, D = args
    t = np.stack([np.asarray(A, float), np.asarray(D, float)], axis=-1)
    cp = sum(r["uncertainty"] for r in cp_rank_sweep(t, range(10, 21), seed=SEED))
    tk = sum(r["uncertainty"] for r in tucker_rank_sweep(t, range(1, 21), third_mode_rank=2, seed=SEED))
    return cp, tk


def official_scores(args):
    s1, s2 = args
    cp = sum(r["uncertainty"] for r in cp_rank_sweep(build_cp_tensor(s1, s2), range(10, 21), seed=SEED))
    tk = sum(r["uncertainty"] for r in tucker_rank_sweep(build_tucker_tensor(s1, s2), range(1, 21), third_mode_rank=2, seed=SEED))
    return cp, tk


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln); out[str(r["id"])] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--max-n", type=int, default=0); ap.add_argument("--check", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    O = a.data_dir; out = O / "experiment_0b" / "gate"
    for cell in a.cells:
        C = Cell(O, cell, "cal"); te = C.rows["test"]; ids = list(C.split["test"])
        s1 = load(O / f"stage_b_s1_semantic_{cell}.jsonl"); s2 = load(O / f"stage_b_s2_claim_{cell}.jsonl")
        sel = np.arange(len(ids))
        if a.max_n and a.max_n < len(ids): sel = np.sort(np.random.default_rng(7).choice(len(ids), a.max_n, replace=False))
        with Pool(a.workers) as p:
            chk = p.map(official_scores, [(s1[ids[i]], s2[ids[i]]) for i in sel[: a.check]])
            res = p.map(sem_scores, [(s1[ids[i]]["matrix_agreement"], s1[ids[i]]["matrix_disagreement"]) for i in sel], chunksize=8)
        stored_cp, stored_tk = C.rawCP[te][sel], C.rawTK[te][sel]
        print(f"[{cell}] reproducibility (first {len(chk)} queries): max|CP diff|={max(abs(c[0]-stored_cp[k]) for k, c in enumerate(chk)):.2e}, max|Tucker diff|={max(abs(c[1]-stored_tk[k]) for k, c in enumerate(chk)):.2e}", flush=True)
        cps, tks = np.array([r[0] for r in res]), np.array([r[1] for r in res])
        U = {"CPsem": cps, "Tsem": tks, "Sumsem": cps + tks}
        ref = {"B3-Sum": stored_cp + stored_tk, "CP(4-slice)": stored_cp, "Tucker(S1+S2 agr)": stored_tk}
        rng = np.random.default_rng(1); n = len(sel); idxs = [rng.integers(0, n, n) for _ in range(a.n_boot)]; rows = []
        for thr in (90, 70):
            acc = C.acc[thr][te][sel]
            for m, u in {**U, **ref}.items():
                row = {"cell": cell, "thr": thr, "method": m, "n": n, "auroc": official_auroc(u, acc, C.N)}
                if m in U:
                    bs = np.array([official_auroc(u[i], acc[i], C.N) for i in idxs]); rb = np.array([official_auroc(ref["B3-Sum"][i], acc[i], C.N) for i in idxs])
                    row["d_B3Sum"] = row["auroc"] - official_auroc(ref["B3-Sum"], acc, C.N); row["lo"], row["hi"] = np.percentile(bs - rb, [2.5, 97.5])
                rows.append(row)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("cell", "thr", "method", "n", "auroc"), k))
        with open(out / f"tsem_{cell}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"[{cell}] done", flush=True)


if __name__ == "__main__":
    main()
