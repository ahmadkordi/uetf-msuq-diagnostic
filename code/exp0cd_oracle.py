#!/usr/bin/env python3
"""Experiments 0C (oracle heterogeneity) and 0D (permutation null), v9 S16. EVALUATION-ONLY
(uses gold-derived U_N_star); analysis ids = train+val only, never test.

0C. For each query: d_k = |tildeU_k - U_N_star|, k*(q) = argmin_k d_k (ties share credit 1/|T|).
    Reported per cell:
      * p_k = share of queries won by estimator k, 'unique wins' (sole argmin), tie fraction
      * normalised entropy H/log K of p_k  (1 = wins spread evenly, 0 = one estimator always wins)
      * MAE_k = mean_q d_k; best single estimator; naive U-Mean (mean of tildeU over k);
        ORACLE MAE (mean_q min_k d_k); headroom = MAE_best_single - MAE_oracle.
    NOTE: oracle MAE is a per-query minimum over K noisy candidates, so it is small even with
    no real structure -- that is exactly why 0D exists; read 0C only against 0D.
0D. Permutation null: shuffle U_N_star across queries (B perms, default 1000), recompute
    entropy / headroom / (optionally) the within-query rank agreement of z_SSL components
    from ssl_<variant>_<cell>.jsonl at eta in {0, 0.5, 1}. Report null mean, 95% interval,
    z-score and two-sided empirical p = (1 + #{|null-mean|>=|obs-mean|}) / (B+1).
Optional --threshold T recomputes U_N_star from stored per-response judge scores
(correct = score > T) instead of using the stored (threshold-90) U_N_star: sensitivity run.
Optional --exclude NAME ... drops estimators from the pool (e.g. s3_self_eval.U_p).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln); out[str(r["id"])] = r
    return out


def read_ids(paths):
    s = set()
    for p in paths: s |= {l.strip() for l in Path(p).read_text().splitlines() if l.strip()}
    return s


def wins_stats(d, tol=1e-12):
    """d: Q x K abs errors -> share p (K,), unique-win share (K,), tie fraction, normalised entropy."""
    m = d.min(1, keepdims=True)
    T = d <= m + tol
    cnt = T.sum(1, keepdims=True)
    p = (T / cnt).mean(0)
    uniq = ((cnt == 1) & T).mean(0)
    pp = p[p > 0]
    H = float(-(pp * np.log(pp)).sum() / np.log(d.shape[1]))
    return p, uniq, float((cnt > 1).mean()), H


def headroom(d):
    mae = d.mean(0)
    return float(mae.min() - d.min(1).mean())


def within_rho_mean(C_ranks_centered, ZA):
    rz = stats.rankdata(ZA, axis=1); rz = rz - rz.mean(1, keepdims=True)
    den = np.sqrt((C_ranks_centered ** 2).sum(1) * (rz ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (C_ranks_centered * rz).sum(1) / den
    return float(np.nanmean(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output"))
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ssl-variant", default=None, choices=[None, "cal", "raw"])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    O = a.data_dir; D = O / "experiment_0b"; Sp = O / "experiment_0a" / "splits"
    cells = a.cells or sorted(p.name[len("stage_g_trainfit_"):-6] for p in D.glob("stage_g_trainfit_*.jsonl"))
    out_lines, summary = [], {}
    for cell in cells:
        keep = read_ids([Sp / f"{cell}.train_ids.txt", Sp / f"{cell}.val_ids.txt"])
        cal = load(D / f"stage_g_trainfit_{cell}.jsonl"); corr = load(O / f"stage_h_correctness_{cell}.jsonl")
        ssl = load(D / f"ssl_{a.ssl_variant}_{cell}.jsonl") if a.ssl_variant else None
        ids = sorted(set(cal) & set(corr) & keep & (set(ssl) if ssl else keep))
        def ustar(i):
            if a.threshold is None: return corr[i].get("U_N_star")
            pr = corr[i]["per_response"]; return 1 - sum(1 for r in pr if r["score"] > a.threshold) / len(pr)
        ids = [i for i in ids if ustar(i) is not None]
        names = [n for n in sorted(cal[ids[0]]["calibrated"]) if n not in a.exclude]
        if ssl: names = [n for n in names if all(n in ssl[i]["hat_a_k"] for i in ids[:50])]
        Ut = np.array([[cal[i]["calibrated"][n] for n in names] for i in ids]); U = np.array([ustar(i) for i in ids])
        d = np.abs(Ut - U[:, None]); K = len(names)
        p, uniq, tie, H = wins_stats(d); hr = headroom(d)
        mae = d.mean(0); mae_mean = float(np.abs(Ut.mean(1) - U).mean()); mae_or = float(d.min(1).mean())
        comps = {}
        if ssl:
            HA = np.array([[ssl[i]["hat_a_k"][n] for n in names] for i in ids]); HS = np.array([[ssl[i]["hat_s_k"][n] for n in names] for i in ids])
            ZA = 1 - d
            for e in (0.0, 0.5, 1.0):
                C = e * HA + (1 - e) * HS; rc = stats.rankdata(C, axis=1); rc = rc - rc.mean(1, keepdims=True)
                comps[e] = (rc, within_rho_mean(rc, ZA))
        rng = np.random.default_rng(a.seed)
        nullH, nullHr = np.empty(a.n_perm), np.empty(a.n_perm); nullP = np.zeros(K); nullW = {e: np.empty(a.n_perm) for e in comps}
        for b in range(a.n_perm):
            Up = U[rng.permutation(len(U))]; dp = np.abs(Ut - Up[:, None])
            _p, _u, _t, nullH[b] = wins_stats(dp); nullP += _p / a.n_perm; nullHr[b] = headroom(dp)
            for e, (rc, _) in comps.items(): nullW[e][b] = within_rho_mean(rc, 1 - dp)
        def tst(obs, null):
            z = (obs - null.mean()) / (null.std() + 1e-12); pv = (1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (len(null) + 1)
            return f"obs={obs:+.4f} null={null.mean():+.4f} [{np.percentile(null,2.5):+.4f},{np.percentile(null,97.5):+.4f}] z={z:+.1f} p={pv:.4f}"
        start = len(out_lines)
        thr = "stored(90)" if a.threshold is None else f"{a.threshold:g}"
        out_lines.append(f"\n== {cell}  nQ={len(ids)} K={K} threshold={thr} excluded={a.exclude} ==")
        out_lines.append(f"  0C entropy H/logK: {tst(H, nullH)}")
        out_lines.append(f"  0C headroom (MAE best single - MAE oracle): {tst(hr, nullHr)}")
        out_lines.append(f"  MAE: best single {names[int(mae.argmin())]}={mae.min():.4f} | U-Mean={mae_mean:.4f} | oracle={mae_or:.4f} | worst={mae.max():.4f} | tie-fraction={tie:.3f}")
        order = np.argsort(-p)
        out_lines.append("  win share (unique-win share) top5: " + "; ".join(f"{names[j]} {p[j]:.3f} ({uniq[j]:.3f})" for j in order[:5]))
        out_lines.append("  win share bottom3: " + "; ".join(f"{names[j]} {p[j]:.3f}" for j in order[-3:]))
        ex = p - nullP; eo = np.argsort(-ex)
        out_lines.append("  EXCESS win share over permutation null (obs - null), top5: " + "; ".join(f"{names[j]} {ex[j]:+.3f} (obs {p[j]:.3f}, null {nullP[j]:.3f})" for j in eo[:5]))
        out_lines.append("  EXCESS bottom3: " + "; ".join(f"{names[j]} {ex[j]:+.3f}" for j in eo[-3:]))
        for e, (_, obs) in comps.items():
            out_lines.append(f"  0D within-query rank agreement z@eta={e:g} ({a.ssl_variant}): {tst(obs, nullW[e])}")
        summary[cell] = dict(n=len(ids), K=K, entropy=H, headroom=hr, mae_best=float(mae.min()), mae_oracle=mae_or, mae_umean=mae_mean,
                             win_share=dict(zip(names, map(float, p))), null_win_share=dict(zip(names, map(float, nullP))), unique_share=dict(zip(names, map(float, uniq))), tie_fraction=tie,
                             null_entropy=[float(nullH.mean()), float(np.percentile(nullH, 2.5)), float(np.percentile(nullH, 97.5))],
                             null_headroom=[float(nullHr.mean()), float(np.percentile(nullHr, 2.5)), float(np.percentile(nullHr, 97.5))])
        print("\n".join(out_lines[start:]), flush=True)
    tag = f"{a.tag}" if a.tag else ("thr" + str(int(a.threshold)) if a.threshold is not None else "thr90")
    (D / f"exp0cd_{tag}.json").write_text(json.dumps(summary, indent=2)); (D / f"exp0cd_{tag}.txt").write_text("\n".join(out_lines))


if __name__ == "__main__":
    main()
