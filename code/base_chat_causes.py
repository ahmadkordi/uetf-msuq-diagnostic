#!/usr/bin/env python3
"""Read-only cause tests for the Llama-2-13B base-vs-chat differences (CPU only, no model calls). New output files only.
T1  response / judge-score statistics per cell (length, empty, judge-score bands, share of responses in the 70-90 band).
T2  accuracy-matched AUROC: chat test queries are re-weighted so that their per-query-accuracy distribution equals the base one
    (and vice versa); asks whether the AUROC difference survives when the accuracy (prevalence) mix is the same.
Usage (from /mnt/src): python base_chat_causes.py [--out /mnt/output/experiment_0b/base_vs_chat]"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
DS = ["coqa", "hotpotqa", "nqopen", "truthfulqa"]; M = [("B2", "B2"), ("B3-Sum", "B3_Sum"), ("B6", "B6")]
EDGES = np.array([0, 1e-9, .2, .5, .8, 1.0001])

def wauroc(u, acc, n, qw):
    t = np.array([int(a * n) for a in acc], float)
    y = np.r_[np.ones(len(u)), np.zeros(len(u))]; w = np.r_[t * qw, (n - t) * qw]; s = np.r_[1 - u, 1 - u]
    k = w > 0
    return float(roc_auc_score(y[k], s[k], sample_weight=w[k])) if len(np.unique(y[k])) == 2 else float("nan")

def bins(a): return np.digitize(a, EDGES[1:-1], right=False)

def matched_w(a_src, a_tgt):  # weights on src queries so that its bin distribution equals tgt's
    bs, bt = bins(a_src), bins(a_tgt); ps = np.bincount(bs, minlength=5) / len(bs); pt = np.bincount(bt, minlength=5) / len(bt)
    r = np.where(ps > 0, pt / np.maximum(ps, 1e-12), 0); return r[bs]

def load(d, cell):
    z = np.load(d / f"uby_{cell}.npz", allow_pickle=True)
    return [str(i) for i in z["ids"]], int(z["N"]), {90: z["acc90"], 70: z["acc70"]}, {m: z[k] for m, k in M}

def txt(x): return x if isinstance(x, str) else x.get("text", "")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("/mnt/output/experiment_0b/base_vs_chat"))
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(); O = a.data_dir; E = O / "experiment_0b"; L = []
    L.append("== T1 response / judge statistics (all queries of the cell) ==")
    L.append(f"{'dataset':11s} {'model':4s} {'chars_mean':>10s} {'empty':>6s} {'score_mean':>10s} {'<=50':>6s} {'50-70':>6s} {'70-90':>6s} {'>90':>6s}")
    for ds in DS:
        for tag, suf in (("base", "Llama2_13b"), ("chat", "Llama2chat_13b")):
            cell = f"{ds}_{suf}"; ch = []; sc = []
            for l in open(O / f"stage_a_{cell}.jsonl", encoding="utf-8"):
                if l.strip(): ch += [len(txt(x)) for x in json.loads(l)["responses"]]
            for l in open(O / f"stage_h_correctness_{cell}.jsonl", encoding="utf-8"):
                pr = json.loads(l).get("per_response")
                if pr: sc += [x["score"] for x in pr]
            ch = np.array(ch); sc = np.array(sc, float)
            L.append(f"{ds:11s} {tag:4s} {ch.mean():10.1f} {(ch==0).mean():6.3f} {sc.mean():10.1f} {(sc<=50).mean():6.3f} {((sc>50)&(sc<=70)).mean():6.3f} {((sc>70)&(sc<=90)).mean():6.3f} {(sc>90).mean():6.3f}")
    L.append("\n== T2 accuracy-matched AUROC (5 accuracy bins: 0, (0,.2], (.2,.5], (.5,.8], (.8,1]); diff = chat - base ==")
    L.append("   'chat->base mix': chat queries re-weighted to the base accuracy mix; 'base->chat mix': the reverse. CI = query bootstrap (weights recomputed).")
    for ds in DS:
        ib, N, accb, Ub = load(E / "stats_extra", f"{ds}_Llama2_13b"); ic, N2, accc, Uc = load(E / "stats_extra_chat", f"{ds}_Llama2chat_13b")
        pb = {i: k for k, i in enumerate(ib)}; o = np.array([pb[i] for i in ic]); n = len(ic); rng = np.random.default_rng(5)
        boots = [rng.integers(0, n, n) for _ in range(a.n_boot)]; one = np.ones(n)
        L.append(f"\n-- {ds} n_test={n}")
        for t in (90, 70):
            ab, ac = accb[t][o], accc[t]
            L.append(f"  tau={t}: acc mix base {np.round(np.bincount(bins(ab),minlength=5)/n,3).tolist()} chat {np.round(np.bincount(bins(ac),minlength=5)/n,3).tolist()}")
            for m, _ in M:
                ub, uc = Ub[m][o], Uc[m]
                raw = wauroc(uc, ac, N2, one) - wauroc(ub, ab, N, one)
                def md(ix, mode):
                    if mode == 0:  # chat reweighted to base mix
                        return wauroc(uc[ix], ac[ix], N2, matched_w(ac[ix], ab[ix])) - wauroc(ub[ix], ab[ix], N, one[ix])
                    return wauroc(uc[ix], ac[ix], N2, one[ix]) - wauroc(ub[ix], ab[ix], N, matched_w(ab[ix], ac[ix]))
                res = []
                for mode in (0, 1):
                    pt = md(np.arange(n), mode); d = np.array([md(ix, mode) for ix in boots]); d = d[np.isfinite(d)]
                    res.append(f"{pt:+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
                L.append(f"    {m:7s} raw diff {raw:+.4f} | chat->base mix {res[0]} | base->chat mix {res[1]}")
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "base_chat_causes.txt").write_text("\n".join(L)); print("\n".join(L))
main()
