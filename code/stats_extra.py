#!/usr/bin/env python3
"""Extra statistics for paper 1 (review fixes). READ-ONLY w.r.t. the frozen protocol:

  * Re-evaluates the FROZEN test configuration (gate_config_<cell>.json) with the same seeds as the original test run.
    Nothing is selected or tuned. The script prints B6/B1/B3-Sum AUROC@90 so you can compare with gate_test_*.txt
    (they must match to 4 decimals; if not, STOP and report).
  * Per cell: AUROC@90/@70 of B1,B1@70,B2,B3-Sum,B3-Min,B4,B5,B5',B6 with 95% query-bootstrap CIs.
  * Per cell: paired deltas for a list of contrasts, bootstrap CI, two-sided bootstrap p, Holm correction
    across the 12 cells within each (contrast, threshold).
  * Pooled mean delta over the 12 cells with a CLUSTER bootstrap: queries are shared across the three generators within a
    dataset, so the same resampled query ids are used for all generators of a dataset (resampling over datasets is NOT done:
    only 4 datasets), reported alongside the per-dataset means.
  * Figures: forest plot of per-cell deltas with CI; heatmap of the mean Spearman correlation between the 15 estimators.

Usage (run from /mnt/src, where gate_eval.py and baseline_auroc.py live):
    python stats_extra.py --data-dir /mnt/output --n-boot 2000 --n-boot-pooled 1000
Outputs go to <data-dir>/experiment_0b/stats_extra/.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # server may lack matplotlib: CSVs are always written, figures are then built elsewhere
    plt = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_eval import Cell, predict_models  # noqa: E402
from baseline_auroc import official_auroc  # noqa: E402

METHODS = ["B1", "B1@70", "B2", "B3-Sum", "B3-Min", "B4", "B5", "B5'", "B6"]
CONTRASTS = [("B6", "B1"), ("B6", "B1@70"), ("B6", "B2"), ("B6", "B3-Sum"), ("B6", "B3-Min"), ("B6", "B4"),
             ("B6", "B5"), ("B6", "B5'"), ("B5", "B2"), ("B3-Sum", "B1"), ("B3-Sum", "B1@70")]
SEEDS = [0, 1, 2]  # same as gate_eval.py default (--seeds 3)


def frozen_predictions(O: Path, cell: str, cfg: dict):
    C = Cell(O, cell, "cal"); ev = C.rows["test"]
    U = {"B1": C.U[ev][:, C.names.index(cfg["B1"]["name"])], "B1@70": C.U[ev][:, C.names.index(cfg["B1@70"]["name"])],
         "B2": C.U[ev].mean(1), "B3-Sum": (C.rawTK + C.rawCP)[ev], "B3-Min": np.minimum(C.rawTK, C.rawCP)[ev]}
    for m in ("B4", "B5", "B5'", "B6"):
        U[m] = predict_models(C, cfg[m]["eta"], cfg[m]["alpha"], ev, SEEDS, {m})[m][0]
    ids = [i for i in C.split["test"]]
    return C, ev, ids, U


def auroc(C, ev, thr, u, idx=None):
    acc = C.acc[thr][ev]
    if idx is not None:
        u, acc = u[idx], acc[idx]
    return official_auroc(u, acc, C.N)


def boot_cell(C, ev, U, n_boot, seed):
    """per-replicate AUROC of every method at both thresholds: dict[(m,thr)] -> array[n_boot]"""
    rng = np.random.default_rng(seed); n = len(ev)
    out = {(m, t): np.empty(n_boot) for m in METHODS for t in (90, 70)}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in METHODS:
            for t in (90, 70):
                out[(m, t)][b] = auroc(C, ev, t, U[m], idx)
    return out


def pval(d):
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1.0 / (len(d) + 1)))


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output"))
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--n-boot-pooled", type=int, default=1000)
    ap.add_argument("--cells", nargs="*", default=None)
    a = ap.parse_args()
    O = a.data_dir; G = O / "experiment_0b" / "gate"; out = O / "experiment_0b" / "stats_extra"; out.mkdir(parents=True, exist_ok=True)
    cells = a.cells or sorted(p.name[len("gate_config_"):-5] for p in G.glob("gate_config_*.json"))
    print("cells:", cells, flush=True)

    store = {}; rows_auc = []; rows_con = []; boots = {}
    for cell in cells:
        cfg = json.loads((G / f"gate_config_{cell}.json").read_text())[cell]
        C, ev, ids, U = frozen_predictions(O, cell, cfg)
        store[cell] = (C, ev, ids, U)
        print(f"[{cell}] reproduction check  B1@90={auroc(C, ev, 90, U['B1']):.4f} B3-Sum@90={auroc(C, ev, 90, U['B3-Sum']):.4f} "
              f"B6@90={auroc(C, ev, 90, U['B6']):.4f}  (compare with gate_test_{cell}.txt)", flush=True)
        np.savez_compressed(out / f"uby_{cell}.npz", ids=np.array(ids), acc90=C.acc[90][ev], acc70=C.acc[70][ev], N=C.N,
                            **{m.replace("'", "p").replace("@", "at").replace("-", "_"): U[m] for m in METHODS})
        bt = boot_cell(C, ev, U, a.n_boot, 1); boots[cell] = bt
        for m in METHODS:
            r = {"cell": cell, "method": m}
            for t in (90, 70):
                r[f"auroc{t}"] = auroc(C, ev, t, U[m]); r[f"lo{t}"] = float(np.nanpercentile(bt[(m, t)], 2.5)); r[f"hi{t}"] = float(np.nanpercentile(bt[(m, t)], 97.5))
            rows_auc.append(r)
        for A, B in CONTRASTS:
            for t in (90, 70):
                d = bt[(A, t)] - bt[(B, t)]; d0 = auroc(C, ev, t, U[A]) - auroc(C, ev, t, U[B])
                rows_con.append({"cell": cell, "contrast": f"{A}-{B}", "thr": t, "delta": d0, "lo": float(np.nanpercentile(d, 2.5)),
                                 "hi": float(np.nanpercentile(d, 97.5)), "p": pval(d)})
        print(f"[{cell}] bootstrap done", flush=True)

    # Holm across cells within (contrast, thr)
    import csv
    for key in sorted({(r["contrast"], r["thr"]) for r in rows_con}):
        sub = [r for r in rows_con if (r["contrast"], r["thr"]) == key]
        for r, adj in zip(sub, holm([r["p"] for r in sub])):
            r["p_holm"] = float(adj)
            r["verdict_raw"] = "win" if r["lo"] > 0 else ("loss" if r["hi"] < 0 else "n.s.")
            r["verdict_holm"] = ("win" if r["delta"] > 0 else "loss") if r["p_holm"] < 0.05 else "n.s."

    def dump(name, rows):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    dump("auroc_ci_by_cell.csv", rows_auc); dump("contrasts_by_cell.csv", rows_con)

    # ---- pooled mean delta with cluster bootstrap (shared query ids across generators of a dataset) ----
    datasets = sorted({c.split("_")[0] for c in cells}); rng = np.random.default_rng(7); pooled = []
    common = {}
    for ds in datasets:
        cs = [c for c in cells if c.split("_")[0] == ds]
        ids = set(store[cs[0]][2])
        for c in cs[1:]: ids &= set(store[c][2])
        common[ds] = sorted(ids)
    pos = {c: {i: k for k, i in enumerate(store[c][2])} for c in cells}
    B = a.n_boot_pooled; need = {(A, B_, t) for A, B_ in CONTRASTS for t in (90, 70)}
    rep = {k: {c: np.empty(B) for c in cells} for k in need}
    for b in range(B):
        draw = {ds: rng.integers(0, len(common[ds]), len(common[ds])) for ds in datasets}
        for c in cells:
            ds = c.split("_")[0]; C, ev, ids, U = store[c]
            idx = np.array([pos[c][common[ds][j]] for j in draw[ds]])
            cache = {(m, t): auroc(C, ev, t, U[m], idx) for m in METHODS for t in (90, 70)}
            for (A, B_, t) in need: rep[(A, B_, t)][c][b] = cache[(A, t)] - cache[(B_, t)]
        if b % 100 == 0: print(f"pooled bootstrap {b}/{B}", flush=True)
    for (A, B_) in CONTRASTS:
        for t in (90, 70):
            per = np.vstack([rep[(A, B_, t)][c] for c in cells]); pm = per.mean(0)
            obs = np.mean([r["delta"] for r in rows_con if r["contrast"] == f"{A}-{B_}" and r["thr"] == t])
            ds_mean = {ds: float(np.mean([r["delta"] for r in rows_con if r["contrast"] == f"{A}-{B_}" and r["thr"] == t and r["cell"].split('_')[0] == ds])) for ds in datasets}
            pooled.append({"contrast": f"{A}-{B_}", "thr": t, "mean_delta_12_cells": float(obs), "lo": float(np.percentile(pm, 2.5)), "hi": float(np.percentile(pm, 97.5)),
                           **{f"mean_{ds}": v for ds, v in ds_mean.items()}})
    dump("pooled_delta_cluster_bootstrap.csv", pooled)

    # ---- figures ----
    for t in ((90, 70) if plt else ()):
        fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
        for ax, (A, B_) in zip(axes, [("B6", "B1" if t == 90 else "B1@70"), ("B6", "B2"), ("B6", "B3-Sum"), ("B3-Sum", "B1" if t == 90 else "B1@70")]):
            sub = [r for r in rows_con if r["contrast"] == f"{A}-{B_}" and r["thr"] == t]
            y = np.arange(len(sub))[::-1]
            ax.errorbar([r["delta"] for r in sub], y, xerr=[[r["delta"] - r["lo"] for r in sub], [r["hi"] - r["delta"] for r in sub]], fmt="o", capsize=2, color="#1f4e79")
            ax.axvline(0, color="k", lw=0.8); ax.set_title(f"{A} - {B_} (AUROC@{t})", fontsize=10)
            ax.set_yticks(y); ax.set_yticklabels([r["cell"] for r in sub], fontsize=7); ax.set_xlabel("paired ΔAUROC, 95% CI")
        fig.tight_layout(); fig.savefig(out / f"forest_tau{t}.png", dpi=200); plt.close(fig)

    # estimator correlation heatmap (train+val, calibrated values, mean Spearman over the 12 cells)
    from scipy.stats import rankdata
    mats = []
    for c in cells:
        C = store[c][0]; rows = np.r_[C.rows["train"], C.rows["val"]]; R = np.apply_along_axis(rankdata, 0, C.U[rows]); mats.append(np.corrcoef(R.T)); names = C.names
    M = np.nanmean(np.stack(mats), 0)
    if plt:
        fig, ax = plt.subplots(figsize=(8.5, 7.5)); im = ax.imshow(M, vmin=-0.2, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=6); ax.set_yticklabels(names, fontsize=6)
        fig.colorbar(im, label="mean Spearman ρ over 12 cells"); fig.tight_layout(); fig.savefig(out / "estimator_corr_heatmap.png", dpi=200); plt.close(fig)
    np.savetxt(out / "estimator_corr.csv", M, delimiter=",", header=",".join(names), comments="")
    print("DONE ->", out)


if __name__ == "__main__":
    main()
