#!/usr/bin/env python3
"""Paired comparison Llama-2-13B base vs chat on the SAME test queries (read-only; evaluation only).

For each dataset the base cell (<ds>_Llama2_13b) and the chat cell (<ds>_Llama2chat_13b) share the same query ids and
the same train/val/test split. Their responses are different (different checkpoint), so the two AUROCs come from
independent samples of responses but the same queries; the paired query-bootstrap resamples the shared query indices
and recomputes both AUROCs on the same resample (difference = chat - base).

Inputs : stats_extra output of each model (uby_<cell>.npz, written by stats_extra.py on the frozen test configuration)
         stage_h_correctness_<cell>.jsonl (for the accuracy table)
Outputs: <out>/base_vs_chat_auroc.csv, base_vs_chat_accuracy.csv, base_vs_chat.txt   (new files only)

Usage (from /mnt/src):
  python compare_base_chat.py --base-dir /mnt/output/experiment_0b/stats_extra \
         --chat-dir /mnt/output/experiment_0b/stats_extra_chat --out /mnt/output/experiment_0b/base_vs_chat
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_auroc import official_auroc  # noqa: E402

DS = ["coqa", "hotpotqa", "nqopen", "truthfulqa"]
KEYS = [("B1", "B1"), ("B1@70", "B1at70"), ("B2", "B2"), ("B3-Sum", "B3_Sum"), ("B3-Min", "B3_Min"),
        ("B4", "B4"), ("B5", "B5"), ("B5'", "B5p"), ("B6", "B6")]


def load(d, cell):
    z = np.load(d / f"uby_{cell}.npz", allow_pickle=True)
    return {"ids": [str(i) for i in z["ids"]], "N": int(z["N"]), "acc": {90: z["acc90"], 70: z["acc70"]},
            "U": {m: z[k] for m, k in KEYS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, required=True); ap.add_argument("--chat-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    for ds in DS:
        B = load(a.base_dir, f"{ds}_Llama2_13b"); C = load(a.chat_dir, f"{ds}_Llama2chat_13b")
        common = sorted(set(B["ids"]) & set(C["ids"]))
        assert len(common) == len(B["ids"]) == len(C["ids"]), f"{ds}: test ids differ between base and chat"
        pb = {i: k for k, i in enumerate(B["ids"])}; pc = {i: k for k, i in enumerate(C["ids"])}
        ib = np.array([pb[i] for i in common]); ic = np.array([pc[i] for i in common]); n = len(common)
        rng = np.random.default_rng(a.seed); boots = [rng.integers(0, n, n) for _ in range(a.n_boot)]
        lines.append(f"\n== {ds}  n_test={n} (same query ids in base and chat) ==")
        lines.append(f"  {'method':7s} {'thr':>3s} {'base':>7s} {'chat':>7s} {'chat-base':>10s}  95% CI")
        for m, _ in KEYS:
            for t in (90, 70):
                ub, uc = B["U"][m][ib], C["U"][m][ic]; ab, ac = B["acc"][t][ib], C["acc"][t][ic]
                ab0 = official_auroc(ub, ab, B["N"]); ac0 = official_auroc(uc, ac, C["N"])
                d = np.array([official_auroc(uc[x], ac[x], C["N"]) - official_auroc(ub[x], ab[x], B["N"]) for x in boots])
                d = d[np.isfinite(d)]; lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
                rows.append(dict(dataset=ds, method=m, thr=t, base=ab0, chat=ac0, diff=ac0 - ab0, lo=lo, hi=hi,
                                 verdict="chat>base" if lo > 0 else ("chat<base" if hi < 0 else "n.s.")))
                lines.append(f"  {m:7s} {t:3d} {ab0:7.4f} {ac0:7.4f} {ac0 - ab0:+10.4f}  [{lo:+.4f}, {hi:+.4f}]")
    # accuracy table (all queries of the cell, per-response judge scores)
    arows = []
    for ds in DS:
        for tag, suf in (("base", "Llama2_13b"), ("chat", "Llama2chat_13b")):
            cell = f"{ds}_{suf}"; a90 = []; a70 = []; z90 = z70 = n = 0
            for line in open(a.data_dir / f"stage_h_correctness_{cell}.jsonl", encoding="utf-8"):
                pr = json.loads(line).get("per_response")
                if not pr: continue
                sc = np.array([x["score"] for x in pr], float); n += 1
                a90.append((sc > 90).mean()); a70.append((sc > 70).mean())
                z90 += int((sc > 90).sum() == 0); z70 += int((sc > 70).sum() == 0)
            arows.append(dict(dataset=ds, model=tag, n_queries=n, acc90=float(np.mean(a90)), acc70=float(np.mean(a70)),
                              none90=z90 / n, none70=z70 / n))
    lines.append("\n== mean sample accuracy and fraction of queries with no correct sample (all queries) ==")
    lines.append(f"  {'dataset':11s} {'model':5s} {'n':>6s} {'acc90':>7s} {'acc70':>7s} {'none90':>7s} {'none70':>7s}")
    for r in arows:
        lines.append(f"  {r['dataset']:11s} {r['model']:5s} {r['n_queries']:6d} {r['acc90']:7.3f} {r['acc70']:7.3f} {r['none90']:7.3f} {r['none70']:7.3f}")
    for name, rr in (("base_vs_chat_auroc.csv", rows), ("base_vs_chat_accuracy.csv", arows)):
        with open(a.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
    (a.out / "base_vs_chat.txt").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
