#!/usr/bin/env python3
"""Read-only: compare Exp 0 results of the Llama-2-13B base cells with the chat cells.
Writes nothing. Run: python exp0_compare_base_chat.py [--dir /mnt/output/experiment_0b]"""
import argparse, json, re
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, default=Path("/mnt/output/experiment_0b"))
a = ap.parse_args()
D = a.dir
DS = ["coqa", "hotpotqa", "nqopen", "truthfulqa"]
PAIR = {"base": "Llama2_13b", "chat": "Llama2chat_13b"}

print("== 0B: mean over estimators of Spearman rho_k (z_SSL vs z_abs), train+val ==")
for ds in DS:
    row = []
    for k, suf in PAIR.items():
        r = json.loads((D / f"exp0b_cal_trainval_{ds}_{suf}.json").read_text())
        pe = r["pooled"]["per_estimator_rho_k_spearman"]
        v = [e["rho"] for e in pe.values() if e.get("rho") is not None]
        npos = sum(1 for x in v if x > 0)
        row.append(f"{k}: mean={np.mean(v):+.3f} pos={npos}/{len(v)} n={r['pooled']['n_queries']}")
    print(f"{ds:11s} " + " | ".join(row))

FILES = {"thr90_cal": ("exp0cd_thr90_cal_v2", "exp0cd_chat_thr90_cal"),
         "thr70": ("exp0cd_thr70", "exp0cd_chat_thr70"),
         "thr90_noUp": ("exp0cd_thr90_noUp", "exp0cd_chat_thr90_noUp")}

def blocks(path):
    out, cur = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"== (\S+)\s", line)
        if m:
            cur = m.group(1); out[cur] = {}; continue
        if cur is None: continue
        m = re.search(r"0C entropy H/logK: obs=(\S+) null=(\S+) .* z=(\S+) p=(\S+)", line)
        if m: out[cur]["ent"] = (m.group(1), m.group(2), m.group(3))
        m = re.search(r"0C headroom .*: obs=(\S+) null=(\S+) .* z=(\S+) p=(\S+)", line)
        if m: out[cur]["hr"] = (m.group(1), m.group(2), m.group(3))
        m = re.search(r"0D .* z@eta=(\S+) \(cal\): obs=(\S+) null=(\S+) .* z=(\S+) p=(\S+)", line)
        if m: out[cur][f"0D_eta{m.group(1)}"] = (m.group(2), m.group(3), m.group(4))
    return out

for tag, (fb, fc) in FILES.items():
    B = blocks(D / (fb + ".txt")); C = blocks(D / (fc + ".txt"))
    print(f"\n== 0C/0D, {tag}  (obs / null / z) ==")
    for ds in DS:
        for k, suf in PAIR.items():
            cell = f"{ds}_{suf}"
            src = B if k == "base" else C
            r = src.get(cell, {})
            f = lambda key: ("/".join(r[key]) if key in r else "n/a")
            print(f"{ds:11s} {k:4s} ent {f('ent'):28s} head {f('hr'):28s} 0D eta0 {f('0D_eta0'):28s} eta1 {f('0D_eta1')}")
