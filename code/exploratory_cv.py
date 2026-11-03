#!/usr/bin/env python3
"""EXPLORATORY analyses for paper 1 (review fixes). Everything here is labelled exploratory.

NO TEST DATA ARE USED. All numbers are out-of-fold (OOF) predictions of K-fold cross-validation over the
pool train+val of each cell (folds are hashed by query id; for CoQA by conversation, so no story leakage).
Hyper-parameters that use labels (eta/gamma selection, B1 selection) are chosen on the training part of each
fold only. The e-CDF calibration, sigma_cons and sigma_stab (stored files) were fitted on the original train
split with no labels; this is label-free and is noted in the paper.

Variants (per cell, per evaluation threshold t in {90, 70}):
  B1cv      best single estimator, chosen on the training folds at the same threshold t (uses labels)
  B2, B3-Sum, B3-Min   fixed rules (no fitting)
  B5(eta)   SSL heuristic w ~ z_SSL, eta fixed at 1 (the most frequently selected value) and 'B5sel' with
            (eta, gamma) selected on the training folds, w ~ z^gamma, gamma in {1,2,4,8,16,inf} (sharpening; inf = argmax)
  B6cv      the gate of the paper retrained inside CV (eta=1, alpha=0.1, gamma=1): control for the CV protocol
  B6g4/g16  gate trained on a sharpened SSL target q ~ zbar^gamma (label-free)
  B7a-<b>   weak supervision with label budget b in {1,5,10,100}% of training queries, ALIGNED target
            z^al_k = 1 - |U~_k - Q(q)|, Q = mid-rank quantile of the query's error rate U*(q)=1-acc_t(q) among labelled
            training queries (LABEL-USING; b=100 is the supervised-target ceiling of the gate)
  B7r-<b>   same with the specification's raw target z^WS_k = 1 - |U~_k - U*(q)|, b in {10,100}
  Loss for labelled queries: KL to (zbar + lam * softmax(z^tgt / T_ws)) / (1 + lam), lam=1, T_ws=0.1
  (this is KL(zbar||p) + lam*KL(softmax(z/T)||p) up to a constant; lam and T_ws are our own choices, fixed a priori, not tuned).

Run from /mnt/src:   python exploratory_cv.py --data-dir /mnt/output --cells <cell> [<cell> ...]
First run ONE cell to see the timing. Outputs: <data-dir>/experiment_0b/exploratory/<cell>.csv and .json
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_eval import Cell, train_gate  # noqa: E402
from baseline_auroc import official_auroc  # noqa: E402

GAMMAS = [1, 2, 4, 8, 16, np.inf]
ETAS = [0.0, 0.5, 1.0]
BUDGETS = [1, 5, 10, 100]
LAM, T_WS, ALPHA0, ETA0 = 1.0, 0.1, 0.1, 1.0


def gkey(i: str, cell: str) -> str:
    return re.sub(r"_\d+$", "", i) if "coqa" in cell.lower() else i


def hfold(key: str, k: int, seed: int) -> int:
    d = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(d[:8], "big") % k


def softmax(z, T):
    e = np.exp((z - z.max(1, keepdims=True)) / T); return e / e.sum(1, keepdims=True)


def sharpen(w, g):
    if np.isinf(g):
        out = np.zeros_like(w); out[np.arange(len(w)), w.argmax(1)] = 1.0; return out
    p = w ** g; return p / p.sum(1, keepdims=True)


def midrank_cdf(ref, x):
    ref = np.sort(ref); lo = np.searchsorted(ref, x, "left"); hi = np.searchsorted(ref, x, "right")
    return (lo + hi) / (2.0 * len(ref))


def auroc(C, rows, thr, u):
    return official_auroc(u, C.acc[thr][rows], C.N)


def targets(C, tr_rows, lab_mask, thr, kind, zbar_tr):
    """per-query training target over estimators; lab_mask marks labelled queries (budget)."""
    q = zbar_tr.copy()
    if kind is None or not lab_mask.any(): return q
    U = C.U[tr_rows][lab_mask]; ustar = 1.0 - C.acc[thr][tr_rows][lab_mask]
    if kind == "al":
        ref = ustar; qq = midrank_cdf(ref, ustar); z = 1.0 - np.abs(U - qq[:, None])
    else:
        z = 1.0 - np.abs(U - ustar[:, None])
    ws = softmax(z, T_WS); q[lab_mask] = (zbar_tr[lab_mask] + LAM * ws) / (1 + LAM)
    return q


def fit_gate(C, X, tr_rows, va_rows, q_tr, q_va, seed):
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-8; Xs = (X - mu) / sd
    g, kv, ep = train_gate(Xs[tr_rows], q_tr, Xs[va_rows], q_va, C.K, seed)
    return lambda rows: g.fwd(Xs[rows])[0]


def run_cell(O: Path, cell: str, k: int, seed: int, loader=Cell):
    C = loader(O, cell, "cal") if loader is Cell else loader
    pool = np.r_[C.rows["train"], C.rows["val"]]; ids = C.split["train"] + C.split["val"]
    folds = np.array([hfold(gkey(i, cell), k, 7) for i in ids]); rng = np.random.default_rng(seed)
    methods = ["B1cv", "B2", "B3-Sum", "B3-Min", "B5", "B5sel", "B6cv", "B6g4", "B6g16"] + [f"B7a-{b}" for b in BUDGETS] + ["B7r-10", "B7r-100"]
    oof = {thr: {m: np.zeros(len(pool)) for m in methods} for thr in (90, 70)}; chosen = []
    t0 = time.time()
    for f in range(k):
        te_m = folds == f; tr_all = pool[~te_m]; te = pool[te_m]
        # inner validation (10% of the training part, hashed) for early stopping on the target KL
        inner = np.array([hfold(gkey(ids[j], cell), 10, 11) == 0 for j in np.where(~te_m)[0]])
        tr, va = tr_all[~inner], tr_all[inner]
        zb = C.zbar(ETA0, ALPHA0); X = C.feats_B6(ETA0)
        lf = {}  # label-free gates (target = zbar sharpened by gamma); independent of the evaluation threshold
        for name, gm in (("B6cv", 1), ("B6g4", 4), ("B6g16", 16)):
            qt = sharpen(zb, gm) if gm != 1 else zb
            pr = fit_gate(C, X, tr, va, qt[tr], qt[va], seed); lf[name] = (pr(te) * C.U[te]).sum(1)
        for thr in (90, 70):
            o = oof[thr]
            k1 = max(range(C.K), key=lambda j: auroc(C, tr_all, thr, C.U[tr_all][:, j])); o["B1cv"][te_m] = C.U[te][:, k1]
            o["B2"][te_m] = C.U[te].mean(1); o["B3-Sum"][te_m] = (C.rawTK + C.rawCP)[te]; o["B3-Min"][te_m] = np.minimum(C.rawTK, C.rawCP)[te]
            z1 = C.z(ETA0); w = z1 / z1.sum(1, keepdims=True); o["B5"][te_m] = (w[te] * C.U[te]).sum(1)
            best = (-1, None)
            for eta in ETAS:
                ze = C.z(eta); we = ze / ze.sum(1, keepdims=True)
                for g in GAMMAS:
                    sc = auroc(C, tr_all, thr, (sharpen(we[tr_all], g) * C.U[tr_all]).sum(1))
                    if sc > best[0]: best = (sc, (eta, g))
            eta, g = best[1]; ze = C.z(eta); we = ze / ze.sum(1, keepdims=True); o["B5sel"][te_m] = (sharpen(we[te], g) * C.U[te]).sum(1)
            chosen.append({"fold": f, "thr": thr, "eta": eta, "gamma": "inf" if np.isinf(g) else g, "b1": C.names[k1]})
            for name in ("B6cv", "B6g4", "B6g16"): o[name][te_m] = lf[name]
            # weak supervision
            for kind, bl in [("al", b) for b in BUDGETS] + [("raw", 10), ("raw", 100)]:
                lab_tr = rng.random(len(tr)) < bl / 100.0; lab_va = rng.random(len(va)) < bl / 100.0
                if bl == 1: lab_tr[rng.choice(len(tr), max(10, int(0.01 * len(tr))), replace=False)] = True
                qtr = targets(C, tr, lab_tr, thr, kind, zb[tr]); qva = targets(C, va, lab_va, thr, kind, zb[va])
                pr = fit_gate(C, X, tr, va, qtr, qva, seed); name = f"B7{'a' if kind == 'al' else 'r'}-{bl}"
                o[name][te_m] = (pr(te) * C.U[te]).sum(1)
        print(f"[{cell}] fold {f + 1}/{k} done  ({time.time() - t0:.0f}s)", flush=True)
    return C, pool, oof, chosen


def boot(C, pool, oof, thr, n_boot, seed):
    rng = np.random.default_rng(seed); n = len(pool); acc = C.acc[thr][pool]; ms = list(oof[thr])
    reps = {m: np.empty(n_boot) for m in ms}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in ms: reps[m][b] = official_auroc(oof[thr][m][idx], acc[idx], C.N)
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=3); ap.add_argument("--n-boot", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); out = a.data_dir / "experiment_0b" / "exploratory"; out.mkdir(parents=True, exist_ok=True)
    import csv
    for cell in a.cells:
        t0 = time.time(); C, pool, oof, chosen = run_cell(a.data_dir, cell, a.folds, a.seed)
        rows = []
        for thr in (90, 70):
            rp = boot(C, pool, oof, thr, a.n_boot, 1)
            for m in oof[thr]:
                r = {"cell": cell, "thr": thr, "method": m, "auroc": auroc(C, pool, thr, oof[thr][m]), "lo": float(np.percentile(rp[m], 2.5)), "hi": float(np.percentile(rp[m], 97.5))}
                for ref in ("B2", "B6cv", "B1cv", "B3-Sum"):
                    d = rp[m] - rp[ref]; r[f"d_{ref}"] = float(auroc(C, pool, thr, oof[thr][m]) - auroc(C, pool, thr, oof[thr][ref]))
                    r[f"lo_{ref}"] = float(np.percentile(d, 2.5)); r[f"hi_{ref}"] = float(np.percentile(d, 97.5))
                rows.append(r)
        with open(out / f"{cell}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        (out / f"{cell}.json").write_text(json.dumps(chosen, indent=1))
        print(f"[{cell}] DONE in {time.time() - t0:.0f}s -> {out / (cell + '.csv')}", flush=True)


if __name__ == "__main__":
    main()
