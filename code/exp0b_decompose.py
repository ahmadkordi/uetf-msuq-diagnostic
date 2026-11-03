#!/usr/bin/env python3
"""Experiment 0B-decomposition: which part of z_SSL (if any) tracks z_abs?

EVALUATION-ONLY diagnostic (uses gold-derived U_N_star). It never feeds back into
ssl_signal.py / bootstrap_stability.py. Analysis ids = train+val only (never test).

For every (query q, estimator k) it builds
    z_abs = 1 - |tildeU_k - U_N_star|
and, from the stored ssl_<variant>_<cell>.jsonl, the components
    a   : Signal A raw          (a_k)
    s   : Signal B raw          (s_k)
    z@eta : eta*hat_a + (1-eta)*hat_s for eta in --etas   (eta=1 -> hat_a, eta=0 -> hat_s)
No re-fit of any scale is done: hat_a/hat_s/a/s are read as stored (sigma's already train-fit).

Three views per component (this separates a query-level factor from estimator-level signal):
  1. across-query rho_k  (as in Exp 0B): Spearman(component_k(q), z_abs_k(q)) over q, per k.
     Reported as mean over k, #k with rho>0 / <0, #k significant (p<0.05, uncorrected).
  2. WITHIN-query: per query, Spearman over the K estimators between component and z_abs;
     mean over queries (+ query-bootstrap CI). Removes any factor that is constant
     across estimators within a query. This is the quantity the gate actually needs.
  3. BETWEEN-query: Spearman over q of (mean_k component) vs (mean_k z_abs) (+ bootstrap CI).
     If this is large while (2) is ~0, the per-estimator rho_k of Exp 0B is carried by a
     query-level factor (e.g. consensus-vs-truth distance), not by estimator discrimination.
Also reports the oracle-free baseline "consensus distance": Spearman over q between mean_k a
and U_N_star (does overall agreement predict hardness?), informational only.

Run (per variant; loops over all cells found):
  python exp0b_decompose.py --data-dir /mnt/output --variant cal
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

LOGGER = logging.getLogger("uetf.exp0b_decompose")


def load_keyed(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                out[str(r["id"])] = r
    return out


def read_ids(paths):
    s = set()
    for p in paths:
        s |= {l.strip() for l in Path(p).read_text().splitlines() if l.strip()}
    return s


def build_arrays(ssl, cal, corr, keep):
    ids = sorted(set(ssl) & set(cal) & set(corr) & keep)
    ids = [i for i in ids if corr[i].get("U_N_star") is not None]
    names = None
    for i in ids:
        n = set(ssl[i]["z_k_SSL"]) & set(cal[i]["calibrated"]) & set(ssl[i]["hat_a_k"]) & set(ssl[i]["hat_s_k"])
        names = n if names is None else names & n
    names = sorted(names)
    Q, K = len(ids), len(names)
    A = np.empty((Q, K)); S = np.empty((Q, K)); HA = np.empty((Q, K)); HS = np.empty((Q, K))
    ZA = np.empty((Q, K)); U = np.empty(Q)
    for qi, i in enumerate(ids):
        u = corr[i]["U_N_star"]; U[qi] = u
        for ki, n in enumerate(names):
            A[qi, ki] = ssl[i]["a_k"][n]; S[qi, ki] = ssl[i]["s_k"][n]
            HA[qi, ki] = ssl[i]["hat_a_k"][n]; HS[qi, ki] = ssl[i]["hat_s_k"][n]
            ZA[qi, ki] = 1.0 - abs(cal[i]["calibrated"][n] - u)
    return ids, names, A, S, HA, HS, ZA, U


def rank_rows(X):
    return np.apply_along_axis(stats.rankdata, 1, X)


def within_query_rho(C, ZA):
    """per-query Spearman over K estimators; NaN where either side is constant."""
    rc, rz = rank_rows(C), rank_rows(ZA)
    rc = rc - rc.mean(1, keepdims=True); rz = rz - rz.mean(1, keepdims=True)
    den = np.sqrt((rc ** 2).sum(1) * (rz ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rc * rz).sum(1) / den
    return r


def boot_mean(x, n_boot, rng):
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None, None, None
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_spearman(a, b, n_boot, rng):
    if len(a) < 3:
        return None, None, None
    r0 = stats.spearmanr(a, b)[0]
    ra, rb = stats.rankdata(a), stats.rankdata(b)  # bootstrap on fixed ranks (Pearson of ranks, approx.)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    xa, xb = ra[idx], rb[idx]
    xa = xa - xa.mean(1, keepdims=True); xb = xb - xb.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = (xa * xb).sum(1) / np.sqrt((xa ** 2).sum(1) * (xb ** 2).sum(1))
    rr = rr[~np.isnan(rr)]
    return float(r0), float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))


def analyse(ids, names, A, S, HA, HS, ZA, U, etas, n_boot, seed):
    rng = np.random.default_rng(seed)
    comps = {"a_raw": A, "s_raw": S}
    for e in etas:
        comps[f"z@eta={e:g}"] = e * HA + (1 - e) * HS
    res = {}
    for cname, C in comps.items():
        rhos, ps = [], []
        for k in range(C.shape[1]):
            if np.ptp(C[:, k]) == 0:
                rhos.append(np.nan); ps.append(np.nan); continue
            r, p = stats.spearmanr(C[:, k], ZA[:, k]); rhos.append(r); ps.append(p)
        rhos, ps = np.array(rhos), np.array(ps)
        ok = ~np.isnan(rhos)
        wq = within_query_rho(C, ZA)
        wm, wlo, whi = boot_mean(wq, n_boot, rng)
        bm, blo, bhi = boot_spearman(C.mean(1), ZA.mean(1), n_boot, rng)
        res[cname] = {
            "across_query": {
                "mean_rho": float(np.nanmean(rhos)) if ok.any() else None,
                "n_pos": int((rhos[ok] > 0).sum()), "n_neg": int((rhos[ok] < 0).sum()),
                "n_sig_pos": int(((rhos > 0) & (ps < 0.05)).sum()), "n_sig_neg": int(((rhos < 0) & (ps < 0.05)).sum()),
                "per_estimator": {n: (None if np.isnan(r) else float(r)) for n, r in zip(names, rhos)},
            },
            "within_query": {"mean": wm, "ci_low": wlo, "ci_high": whi, "n_queries_used": int((~np.isnan(wq)).sum())},
            "between_query": {"rho": bm, "ci_low": blo, "ci_high": bhi},
        }
    cons = stats.spearmanr(A.mean(1), U)[0] if np.ptp(A.mean(1)) > 0 and np.ptp(U) > 0 else None
    return {"n_queries": len(ids), "n_estimators": len(names), "estimators": names,
            "spearman_meanA_vs_U_N_star": None if cons is None else float(cons), "components": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output"))
    ap.add_argument("--variant", default="cal", choices=["cal", "raw"])
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260903)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    O = a.data_dir; D = O / "experiment_0b"; Sp = O / "experiment_0a" / "splits"
    cells = a.cells or sorted(p.name[len(f"ssl_{a.variant}_"):-6] for p in D.glob(f"ssl_{a.variant}_*.jsonl"))
    rows = []
    for cell in cells:
        keep = read_ids([Sp / f"{cell}.train_ids.txt", Sp / f"{cell}.val_ids.txt"])   # never test
        ssl = load_keyed(D / f"ssl_{a.variant}_{cell}.jsonl")
        cal = load_keyed(D / f"stage_g_trainfit_{cell}.jsonl")
        corr = load_keyed(O / f"stage_h_correctness_{cell}.jsonl")
        arr = build_arrays(ssl, cal, corr, keep)
        rep = analyse(*arr, a.etas, a.n_boot, a.seed)
        (D / f"exp0b_decomp_{a.variant}_{cell}.json").write_text(json.dumps(rep, indent=2))
        for c, v in rep["components"].items():
            f = lambda x: "" if x is None else f"{x:+.3f}"
            rows.append([cell, c, rep["n_queries"], f(v["across_query"]["mean_rho"]),
                         f"{v['across_query']['n_pos']}+/{v['across_query']['n_neg']}-",
                         f(v["within_query"]["mean"]), f"[{f(v['within_query']['ci_low'])},{f(v['within_query']['ci_high'])}]",
                         f(v["between_query"]["rho"]), f"[{f(v['between_query']['ci_low'])},{f(v['between_query']['ci_high'])}]"])
    hdr = ["cell", "component", "nQ", "mean_rho_k", "pos/neg", "WITHIN_q", "within_CI", "BETWEEN_q", "between_CI"]
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    lines = ["  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)) for r in [hdr] + rows]
    print("\n".join(lines))
    (D / f"exp0b_decomp_summary_{a.variant}.txt").write_text("\n".join(lines))
    with open(D / f"exp0b_decomp_summary_{a.variant}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([hdr] + rows)


if __name__ == "__main__":
    main()
