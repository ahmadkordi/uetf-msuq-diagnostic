#!/usr/bin/env python3
"""Read-only: build the tau = 70 versions of Tables 3 and 4 and the 0C summary, and compare
them with tau = 90. Writes one Markdown file (default <data-dir>/experiment_0b/tau70/exp0_tau70_tables.md).

Inputs (all read-only):
  <out>/exp0b_decomp_cal_<cell>.json            exp0_tau70.py           (0B, tau = 70)
  experiment_0b/exp0b_decomp_cal_<cell>.json     exp0b_decompose.py      (0B, tau = 90, as in Table 3)
  experiment_0b/exp0cd_<tag70>.txt               exp0cd_oracle.py --threshold 70 --ssl-variant cal
  experiment_0b/exp0cd_<tag90>.txt               exp0cd_oracle.py (tau = 90, cal; as in Table 4)
  experiment_0b/exp0cd_thr70.txt (optional)      the earlier tau = 70 run, used only to check the Llama-2 cells

Run:  python exp0_tau70_tables.py --data-dir /mnt/output
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATASETS = [("coqa", "CoQA"), ("hotpotqa", "HotpotQA"), ("nqopen", "NQ-Open"), ("truthfulqa", "TruthfulQA")]
MODELS = [("Llama2_13b", "Llama-2"), ("llama3.1_8b", "Llama-3.1"), ("Q2.5_14b", "Qwen2.5")]
CELLS = [(f"{d}_{m}", f"{dn} - {mn}") for d, dn in DATASETS for m, mn in MODELS]
NUM = r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)"
STAT = re.compile(rf"obs={NUM} null={NUM} \[{NUM},{NUM}\] z={NUM} p={NUM}")


def parse_cd(path: Path) -> dict:
    out, cur = {}, None
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\s*== (\S+)\s+nQ=(\d+)", line)
        if m:
            cur = m.group(1); out[cur] = {"n": int(m.group(2))}; continue
        if cur is None:
            continue
        s = STAT.search(line)
        if not s:
            continue
        vals = dict(zip(["obs", "null", "lo", "hi", "z", "p"], map(float, s.groups())))
        if "0C entropy" in line:
            out[cur]["ent"] = vals
        elif "0C headroom" in line:
            out[cur]["hr"] = vals
        else:
            m = re.search(r"z@eta=(\S+) \(", line)
            if m:
                out[cur][f"eta{float(m.group(1)):g}"] = vals
    return out


def dec(path: Path) -> dict | None:
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    c = r["components"]
    return {"n": r["n_queries"],
            "A_rho": c["z@eta=1"]["across_query"]["mean_rho"], "A_w": c["z@eta=1"]["within_query"]["mean"],
            "B_rho": c["z@eta=0"]["across_query"]["mean_rho"], "B_w": c["z@eta=0"]["within_query"]["mean"],
            "zero": r.get("frac_zero_correct")}


def f(x, nd=3):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def excess(v: dict | None, alpha=0.01):
    if not v:
        return "n/a", None
    e = v["obs"] - v["null"]
    tag = "" if v["p"] < alpha else " (n.s.)"
    return f"{e:+.3f}{tag}", e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output"))
    ap.add_argument("--dec70-dir", type=Path, default=None, help="default: <data-dir>/experiment_0b/tau70")
    ap.add_argument("--cd70", default="exp0cd_thr70_cal_12cells.txt")
    ap.add_argument("--cd90", default="exp0cd_thr90_cal_v2.txt")
    ap.add_argument("--cd70-old", default="exp0cd_thr70.txt")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    D = a.data_dir / "experiment_0b"
    T = a.dec70_dir or (D / "tau70")
    cd70, cd90, cdold = parse_cd(D / a.cd70), parse_cd(D / a.cd90), parse_cd(D / a.cd70_old)
    L = []
    L.append("# Experiment 0 at tau = 70 (train+val; cal variant)\n")
    L.append(f"Sources: {T}/exp0b_decomp_cal_*.json; {D / a.cd70}; tau = 90 for comparison: {D}/exp0b_decomp_cal_*.json, {D / a.cd90}.\n")

    L.append("## Table 3 at tau = 70 (0B decomposition; A = Signal A alone, eta = 1; B = Signal B alone, eta = 0)\n")
    L.append("| Cell | n | frac acc=0 | A: mean rho_k | A: within-q | B: mean rho_k | B: within-q |")
    L.append("|---|---|---|---|---|---|---|")
    for c, name in CELLS:
        r = dec(T / f"exp0b_decomp_cal_{c}.json")
        if r is None:
            L.append(f"| {name} | missing | | | | | |"); continue
        L.append(f"| {name} | {r['n']} | {r['zero']:.3f} | {f(r['A_rho'])} | {f(r['A_w'])} | {f(r['B_rho'])} | {f(r['B_w'])} |")

    L.append("\n## Table 4 at tau = 70 (0D: within-query rank agreement, observed vs permutation null; excess marked n.s. if p >= 0.01)\n")
    L.append("| Cell | eta=1 obs | eta=1 null | eta=1 excess | z | eta=0 obs | eta=0 null | eta=0 excess | z |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c, name in CELLS:
        r = cd70.get(c, {})
        e1, e0 = r.get("eta1"), r.get("eta0")
        if not e1 or not e0:
            L.append(f"| {name} | missing | | | | | | | |"); continue
        L.append(f"| {name} | {f(e1['obs'])} | {f(e1['null'])} | {excess(e1)[0]} | {e1['z']:+.1f} | "
                 f"{f(e0['obs'])} | {f(e0['null'])} | {excess(e0)[0]} | {e0['z']:+.1f} |")

    L.append("\n## 0C at tau = 70 (entropy of the per-query best estimator and headroom, observed vs null)\n")
    L.append("| Cell | entropy obs | null | z | headroom obs | null | z |")
    L.append("|---|---|---|---|---|---|---|")
    for c, name in CELLS:
        r = cd70.get(c, {})
        en, hr = r.get("ent"), r.get("hr")
        if not en or not hr:
            L.append(f"| {name} | missing | | | | | |"); continue
        L.append(f"| {name} | {en['obs']:.3f} | {en['null']:.3f} | {en['z']:+.1f} | {hr['obs']:.3f} | {hr['null']:.3f} | {hr['z']:+.1f} |")

    L.append("\n## Comparison tau = 90 vs tau = 70\n")
    L.append("| Cell | A within-q 90 / 70 | B within-q 90 / 70 | eta=1 excess 90 / 70 | eta=0 excess 90 / 70 | entropy z 90 / 70 |")
    L.append("|---|---|---|---|---|---|")
    cnt = {"e1_pos90": 0, "e1_pos70": 0, "e0_pos90": 0, "e0_pos70": 0, "ent90": 0, "ent70": 0, "n90": 0, "n70": 0}
    for c, name in CELLS:
        d9, d7 = dec(D / f"exp0b_decomp_cal_{c}.json"), dec(T / f"exp0b_decomp_cal_{c}.json")
        r9, r7 = cd90.get(c, {}), cd70.get(c, {})
        x91, x71 = excess(r9.get("eta1")), excess(r7.get("eta1"))
        x90, x70 = excess(r9.get("eta0")), excess(r7.get("eta0"))
        z9 = r9.get("ent", {}).get("z"); z7 = r7.get("ent", {}).get("z")
        for key, v in (("9", r9), ("7", r7)):
            if v.get("eta1"):
                cnt["n" + key + "0"] += 1
                cnt["e1_pos" + key + "0"] += int(v["eta1"]["obs"] > v["eta1"]["null"] and v["eta1"]["p"] < 0.01)
                cnt["e0_pos" + key + "0"] += int(v["eta0"]["obs"] > v["eta0"]["null"] and v["eta0"]["p"] < 0.01)
            if v.get("ent"):
                cnt["ent" + key + "0"] += int(v["ent"]["obs"] > v["ent"]["null"] and v["ent"]["p"] < 0.01)
        L.append(f"| {name} | {f(d9 and d9['A_w'])} / {f(d7 and d7['A_w'])} | {f(d9 and d9['B_w'])} / {f(d7 and d7['B_w'])} | "
                 f"{x91[0]} / {x71[0]} | {x90[0]} / {x70[0]} | "
                 f"{'n/a' if z9 is None else f'{z9:+.1f}'} / {'n/a' if z7 is None else f'{z7:+.1f}'} |")
    L.append("")
    L.append(f"Cells with excess above the null (p < 0.01): eta = 1: {cnt['e1_pos90']}/{cnt['n90']} at tau = 90, "
             f"{cnt['e1_pos70']}/{cnt['n70']} at tau = 70; eta = 0: {cnt['e0_pos90']}/{cnt['n90']} and {cnt['e0_pos70']}/{cnt['n70']}; "
             f"entropy above the null: {cnt['ent90']} and {cnt['ent70']}.")

    L.append("\n## Check: the new tau = 70 run against the earlier one (Llama-2 cells of Table 18)\n")
    worst = 0.0; nchk = 0
    for c, name in CELLS:
        if c in cdold and c in cd70:
            for k in ("ent", "hr", "eta0", "eta1"):
                if k in cdold[c] and k in cd70[c]:
                    worst = max(worst, abs(cdold[c][k]["obs"] - cd70[c][k]["obs"]), abs(cdold[c][k]["null"] - cd70[c][k]["null"]))
            nchk += 1
    L.append(f"{nchk} cells compared; max |difference| of obs and null = {worst:.2e} (expected 0)." if nchk
             else "earlier file not found or no common cells; check skipped.")

    out = a.out or (T / "exp0_tau70_tables.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
