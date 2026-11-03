#!/usr/bin/env python3
"""Parse gate_eval log (val or test) -> per-cell tables + win/loss counts of paired contrasts (CI excludes 0).
Usage: python summarize_gate.py /mnt/output/experiment_0b/logs/gate_test.log [out_prefix]"""
import re, sys, csv
from pathlib import Path
log = Path(sys.argv[1]).read_text(); pre = sys.argv[2] if len(sys.argv) > 2 else str(Path(sys.argv[1]).with_suffix(""))
cells = {}; cur = None
for ln in log.splitlines():
    m = re.match(r"== (\S+)\s+split=(\w+)", ln)
    if m: cur = m.group(1); cells[cur] = {"auc": {}, "pair": {}}; continue
    if cur is None: continue
    m = re.match(r"\s+(B[\w'@-]+)\s+([\d.]+)\s+([\d.]+)\s*(.*)", ln)
    if m and not ln.strip().startswith("paired"): cells[cur]["auc"][m.group(1)] = (float(m.group(2)), float(m.group(3)), m.group(4).strip()); continue
    m = re.match(r"\s+paired (\S+) - (\S+) @ (\d+): ([+-][\d.]+) \[95% CI ([+-][\d.]+), ([+-][\d.]+)\]", ln)
    if m: cells[cur]["pair"][(m.group(1), m.group(2), int(m.group(3)))] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
meths = ["B1", "B1@70", "B2", "B3-Sum", "B3-Min", "B4", "B5", "B5'", "B6"]
rows = []
with open(pre + "_auroc.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cell", "thr"] + meths)
    for c, d in cells.items():
        for i, thr in enumerate((90, 70)): w.writerow([c, thr] + [f"{d['auc'][m][i]:.4f}" if m in d["auc"] else "" for m in meths])
out = []
for thr in (90, 70):
    out.append(f"\n### AUROC@{thr} (official protocol)\n| cell | " + " | ".join(meths) + " |\n|" + "---|" * (len(meths) + 1))
    for c, d in cells.items(): out.append(f"| {c} | " + " | ".join(f"{d['auc'][m][0 if thr == 90 else 1]:.3f}" if m in d["auc"] else "" for m in meths) + " |")
out.append("\n### Paired contrasts: cells where CI excludes 0 (win / n.s. / loss), out of %d" % len(cells))
for A, B in (("B6", "B1"), ("B6", "B1@70"), ("B6", "B2"), ("B6", "B3-Sum"), ("B6", "B5"), ("B6", "B5'"), ("B5", "B2")):
    for thr in (90, 70):
        w_ = l_ = n_ = 0; ds = []
        for c, d in cells.items():
            k = (A, B, thr)
            if k not in d["pair"]:
                continue
            x, lo, hi = d["pair"][k]; ds.append(x)
            if lo > 0: w_ += 1
            elif hi < 0: l_ += 1
            else: n_ += 1
        if ds: out.append(f"{A} - {B} @ {thr}: win {w_} / n.s. {n_} / loss {l_}; mean delta {sum(ds)/len(ds):+.4f}")
Path(pre + "_summary.md").write_text("\n".join(out)); print("\n".join(out))
