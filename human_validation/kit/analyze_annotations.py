#!/usr/bin/env python3
"""Analyse the human annotation of judged responses.

  python analyze_annotations.py --dir annotation_out --pilot   # after the pilot round
  python analyze_annotations.py --dir annotation_out           # after the main round

Main round, first call : inter-annotator agreement + adjudication.xlsx (disagreements).
                         No judge information is printed at this stage.
Main round, second call: (after adjudication.xlsx is filled) judge-vs-human agreement at
                         each threshold, per dataset and per generator, a threshold sweep
                         and a paper-ready summary in <dir>/results/.

All population estimates are weighted: every sampled response stands for
cell_pop / n_valid responses of its dataset x score-band x generator cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from annot_common import (LABELS, TEXT_COLUMNS, cohen_kappa, normalise_labels, read_sheet,
                          write_sheet)

KEY_NAME = "KEY_do_not_open.csv"
METRICS = ["accuracy", "kappa", "precision", "recall", "specificity", "false_pos_rate",
           "human_correct_rate", "judge_correct_rate"]


# --------------------------------------------------------------------------- helpers
def load_labels(directory: Path, stem: str, who: str) -> pd.DataFrame:
    df = read_sheet(directory / stem)
    if df is None:
        sys.exit(f"[error] {directory / stem}.xlsx (or .csv) not found.")
    for col in ("item_id", "label"):
        if col not in df.columns:
            sys.exit(f"[error] {stem}: column '{col}' is missing.")
    df["label"] = normalise_labels(df["label"], stem)
    for col in ("issue_tag", "notes"):
        if col not in df.columns:
            df[col] = ""
    if df["item_id"].duplicated().any():
        sys.exit(f"[error] {stem}: duplicated item_id values - rows were copied or edited.")
    keep = ["item_id", "label", "issue_tag", "notes"]
    return df.rename(columns={c: f"{c}_{who}" for c in keep[1:]}), \
        [c for c in df.columns if c in ["dataset"] + TEXT_COLUMNS]


def sheet_format(directory: Path, stem: str) -> str:
    """Write follow-up sheets in the same format as the annotation sheets."""
    return "xlsx" if (directory / f"{stem}.xlsx").exists() else "csv"


def boot_ci(values: np.ndarray, level: float = 0.95):
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanpercentile(values, lo, axis=0), np.nanpercentile(values, hi, axis=0)


def iaa_row(name: str, a: pd.Series, b: pd.Series, rng, n_boot: int) -> dict:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    both = a.isin(["CORRECT", "INCORRECT"]) & b.isin(["CORRECT", "INCORRECT"])
    ab, bb = a[both].reset_index(drop=True), b[both].reset_index(drop=True)
    row = {
        "scope": name, "n": len(a),
        "agreement_3way": float((a == b).mean()) if len(a) else np.nan,
        "kappa_3way": cohen_kappa(a, b),
        "n_binary": int(both.sum()),
        "agreement_binary": float((ab == bb).mean()) if len(ab) else np.nan,
        "kappa_binary": cohen_kappa(ab, bb),
        "unclear_A": int((a == "UNCLEAR").sum()), "unclear_B": int((b == "UNCLEAR").sum()),
    }
    if len(ab) >= 10 and n_boot:
        idx = rng.integers(0, len(ab), size=(n_boot, len(ab)))
        av, bv = (ab == "CORRECT").to_numpy(), (bb == "CORRECT").to_numpy()
        ks = np.empty(n_boot)
        for r in range(n_boot):
            x, y = av[idx[r]], bv[idx[r]]
            po = (x == y).mean()
            pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
            ks[r] = (po - pe) / (1 - pe) if pe < 1 else np.nan
        lo, hi = boot_ci(ks)
        row["kappa_binary_lo"], row["kappa_binary_hi"] = float(lo), float(hi)
    return row


def conf_by_group(g: np.ndarray, n_groups: int, h: np.ndarray, j: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
    """Weighted confusion matrices, shape (n_groups, 2, 2) indexed [group, human, judge]."""
    code = g * 4 + h * 2 + j
    return np.bincount(code, weights=w, minlength=n_groups * 4).reshape(n_groups, 2, 2)


def metrics(conf: np.ndarray) -> dict:
    """Metrics from confusion matrices with leading batch dims; [.., human, judge]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tn, fp, fn, tp = conf[..., 0, 0], conf[..., 0, 1], conf[..., 1, 0], conf[..., 1, 1]
        total = tn + fp + fn + tp
        acc = (tp + tn) / total
        ph, pj = (tp + fn) / total, (tp + fp) / total
        pe = ph * pj + (1 - ph) * (1 - pj)
        kappa = np.where(pe < 1 - 1e-12, (acc - pe) / (1 - pe), np.nan)
        return {
            "accuracy": acc, "kappa": kappa,
            "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp), "false_pos_rate": fp / (tn + fp),
            "human_correct_rate": ph, "judge_correct_rate": pj,
        }


def judge_correct(score: np.ndarray, tau: float, rule: str) -> np.ndarray:
    return (score > tau if rule == ">" else score >= tau).astype(np.int64)


def nanmean0(x: np.ndarray, axis: int) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


class Design:
    """Arrays describing the valid (labelled) sample."""

    def __init__(self, df: pd.DataFrame, meta: dict, sweep: list):
        self.datasets = [d for d in meta["datasets"] if d in set(df["dataset"])]
        self.models = [m for m in meta["models"] if m in set(df["model"])]
        self.bands = list(meta["band_labels"])
        self.taus = [float(t) for t in meta["thresholds"]]
        self.sweep = sweep
        self.rule = meta["correct_if"]
        self.ds = df["dataset"].map({d: i for i, d in enumerate(self.datasets)}).to_numpy()
        self.md = df["model"].map({m: i for i, m in enumerate(self.models)}).to_numpy()
        self.band = df["band"].to_numpy().astype(np.int64)
        self.w = df["weight"].to_numpy(dtype=float)
        fam = set(meta.get("judge_family_models") or [])
        self.fam = df["model"].isin(fam).to_numpy().astype(np.int64)
        self.has_family = bool(fam) and 0 < self.fam.sum() < len(df)

    def compute(self, h: np.ndarray, score: np.ndarray) -> dict:
        """All statistics for one (re)sample; returns {name: ndarray}."""
        D, M = len(self.datasets), len(self.models)
        out = {}
        per_ds, pooled, per_md, fam = [], [], [], []
        for tau in self.taus:
            j = judge_correct(score, tau, self.rule)
            c_ds = conf_by_group(self.ds, D, h, j, self.w)
            per_ds.append(metrics(c_ds))
            pooled.append(metrics(c_ds.sum(axis=0)))
            per_md.append(metrics(conf_by_group(self.md, M, h, j, self.w)))
            if self.has_family:
                fam.append(metrics(conf_by_group(self.fam, 2, h, j, self.w)))
        for name in METRICS:
            out[f"ds.{name}"] = np.stack([m[name] for m in per_ds])        # (T, D)
            out[f"pooled.{name}"] = np.stack([m[name] for m in pooled])    # (T,)
            out[f"macro.{name}"] = nanmean0(out[f"ds.{name}"], axis=1)     # (T,)
            out[f"model.{name}"] = np.stack([m[name] for m in per_md])     # (T, M)
            if self.has_family:
                f = np.stack([m[name] for m in fam])                       # (T, 2)
                out[f"family_diff.{name}"] = f[:, 1] - f[:, 0]
        # difference between the highest and the lowest threshold
        out["diff.macro_kappa"] = out["macro.kappa"][-1] - out["macro.kappa"][0]
        out["diff.pooled_kappa"] = out["pooled.kappa"][-1] - out["pooled.kappa"][0]
        out["diff.ds_kappa"] = out["ds.kappa"][-1] - out["ds.kappa"][0]
        # human-correct rate inside each judge-score band
        B = len(self.bands)
        num = np.bincount(self.ds * B + self.band, weights=self.w * h, minlength=D * B)
        den = np.bincount(self.ds * B + self.band, weights=self.w, minlength=D * B)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["band.rate"] = (num / den).reshape(D, B)
        out["band.macro_rate"] = nanmean0(out["band.rate"], axis=0)
        # threshold sweep
        sw_macro, sw_pooled, sw_acc, sw_ds = [], [], [], []
        for tau in self.sweep:
            j = judge_correct(score, tau, self.rule)
            c_ds = conf_by_group(self.ds, D, h, j, self.w)
            m_ds, m_all = metrics(c_ds), metrics(c_ds.sum(axis=0))
            sw_ds.append(m_ds["kappa"])
            sw_macro.append(nanmean0(m_ds["kappa"], axis=0))
            sw_pooled.append(m_all["kappa"])
            sw_acc.append(m_all["accuracy"])
        out["sweep.ds_kappa"] = np.stack(sw_ds)
        out["sweep.macro_kappa"] = np.array(sw_macro)
        out["sweep.pooled_kappa"] = np.array(sw_pooled)
        out["sweep.pooled_accuracy"] = np.array(sw_acc)
        return out


def bootstrap(design: Design, h: np.ndarray, score: np.ndarray, cell: np.ndarray,
              n_boot: int, rng) -> dict:
    """Stratified bootstrap: resample responses within dataset x band x generator cells."""
    n = len(h)
    idx = np.empty((n_boot, n), dtype=np.int64)
    for c in np.unique(cell):
        pos = np.flatnonzero(cell == c)
        idx[:, pos] = rng.choice(pos, size=(n_boot, len(pos)), replace=True)
    reps: dict = {}
    for r in range(n_boot):
        res = design.compute(h[idx[r]], score[idx[r]])
        for k, v in res.items():
            reps.setdefault(k, []).append(v)
    return {k: boot_ci(np.stack(v)) for k, v in reps.items()}


def f3(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def with_ci(x, lo, hi) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.3f} [{f3(lo)}, {f3(hi)}]"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- pilot
def run_pilot(directory: Path, rng, n_boot: int) -> None:
    a, text_cols = load_labels(directory, "pilot_A", "A")
    b, _ = load_labels(directory, "pilot_B", "B")
    m = a.merge(b[["item_id", "label_B", "issue_tag_B", "notes_B"]], on="item_id")
    done = m[(m["label_A"] != "") & (m["label_B"] != "")]
    print(f"Pilot: {len(done)} of {len(m)} items labelled by both annotators.")
    if done.empty:
        return
    rows = [iaa_row("ALL", done["label_A"], done["label_B"], rng, n_boot)]
    for ds, block in done.groupby("dataset"):
        rows.append(iaa_row(ds, block["label_A"], block["label_B"], rng, 0))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    dis = done[done["label_A"] != done["label_B"]]
    cols = ["item_id"] + text_cols + ["label_A", "issue_tag_A", "notes_A",
                                      "label_B", "issue_tag_B", "notes_B"]
    path = write_sheet(directory / "pilot_disagreements", dis[cols], {"agreed_rule": None},
                       sheet_format(directory, "pilot_A"))
    print(f"\n{len(dis)} disagreements written to {path}.\n"
          "Discuss them, add the resulting rules to the guidelines, then freeze the "
          "guidelines before the main round. Pilot items are not part of the analysis.")


# --------------------------------------------------------------------------- main round
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path, help="output_dir of the sampling step")
    ap.add_argument("--pilot", action="store_true", help="analyse the pilot round only")
    ap.add_argument("--provisional", action="store_true",
                    help="do not wait for adjudication: drop unresolved disagreements")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="continue although some rows are not labelled yet")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--sweep-step", type=float, default=5.0)
    args = ap.parse_args()
    d = args.dir
    rng = np.random.default_rng(args.seed)

    if args.pilot:
        run_pilot(d, rng, args.n_boot)
        return

    meta = json.loads((d / "sampling_meta.json").read_text(encoding="utf-8"))
    key = pd.read_csv(d / KEY_NAME, dtype={"item_id": str, "query_id": str})
    key = key[key["set"] == "main"].copy()
    a, text_cols = load_labels(d, "annotator_A", "A")
    b, _ = load_labels(d, "annotator_B", "B")

    unknown = set(a["item_id"]) - set(key["item_id"])
    if unknown:
        sys.exit(f"[error] annotator_A has {len(unknown)} item ids that are not in the key. "
                 "Is this the right directory?")
    df = key.merge(a.drop(columns=["dataset"], errors="ignore"), on="item_id", how="left").merge(
        b[["item_id", "label_B", "issue_tag_B", "notes_B"]], on="item_id", how="left")
    for col in ("label_A", "label_B", "issue_tag_A", "issue_tag_B", "notes_A", "notes_B"):
        df[col] = df[col].fillna("")
    in_b = df["item_id"].isin(set(b["item_id"]))

    miss_a = int((df["label_A"] == "").sum())
    miss_b = int(((df["label_B"] == "") & in_b).sum())
    if miss_a or miss_b:
        msg = f"unlabelled rows: annotator_A {miss_a}, annotator_B {miss_b}"
        if not args.allow_incomplete:
            sys.exit(f"[error] {msg}. Finish the sheets or pass --allow-incomplete.")
        print(f"[warn] {msg}; they are left out.")
    df = df[df["label_A"] != ""].copy()
    both = in_b.reindex(df.index) & (df["label_B"] != "")

    # ---- 1. inter-annotator agreement (no judge information involved)
    out = d / "results"
    out.mkdir(exist_ok=True)
    dbl = df[both]
    iaa_rows = [iaa_row("ALL", dbl["label_A"], dbl["label_B"], rng, args.n_boot)]
    for ds in meta["datasets"]:
        block = dbl[dbl["dataset"] == ds]
        if len(block):
            iaa_rows.append(iaa_row(ds, block["label_A"], block["label_B"], rng, args.n_boot))
    iaa = pd.DataFrame(iaa_rows)
    iaa.to_csv(out / "inter_annotator_agreement.csv", index=False)
    print(f"Inter-annotator agreement ({len(dbl)} double-annotated items):")
    print(iaa.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 2. adjudication of disagreements (still blind to judge and generator)
    dis = dbl[dbl["label_A"] != dbl["label_B"]]
    df["final"] = np.where(both & (df["label_A"] != df["label_B"]), "", df["label_A"])
    df["label_source"] = np.where(both, "agreed", "single_annotator")
    if len(dis):
        adj = read_sheet(d / "adjudication")
        if adj is None:
            cols = ["item_id"] + text_cols + ["label_A", "issue_tag_A", "notes_A",
                                              "label_B", "issue_tag_B", "notes_B"]
            order = rng.permutation(len(dis))
            path = write_sheet(d / "adjudication", dis.iloc[order][cols],
                               {"final_label": LABELS, "final_notes": None},
                               sheet_format(d, "annotator_A"))
            print(f"\n{len(dis)} disagreements written to {path}.\n"
                  "Resolve them together (fill final_label) WITHOUT opening the key, "
                  "then run this command again.")
            if not args.provisional:
                return
            adj = pd.DataFrame({"item_id": [], "final_label": []})
        adj["final_label"] = normalise_labels(adj["final_label"], "adjudication")
        resolved = dict(zip(adj["item_id"], adj["final_label"]))
        need = set(dis["item_id"])
        open_ids = {i for i in need if resolved.get(i, "") == ""}
        if open_ids and not args.provisional:
            sys.exit(f"[error] adjudication: {len(open_ids)} of {len(need)} disagreements "
                     "have no final_label. Fill them or pass --provisional.")
        mask = df["item_id"].isin(need)
        df.loc[mask, "final"] = df.loc[mask, "item_id"].map(lambda i: resolved.get(i, ""))
        df.loc[mask, "label_source"] = "adjudicated"
        if open_ids:
            print(f"[warn] provisional run: {len(open_ids)} unresolved disagreements dropped.")
    df = df[df["final"] != ""].copy()

    final_cols = ["item_id", "dataset", "model", "query_id", "sample_idx", "judge_score",
                  "band_label", "label_A", "label_B", "final", "label_source",
                  "issue_tag_A", "issue_tag_B", "cell_pop"]
    df[final_cols].to_csv(out / "final_labels.csv", index=False)

    # ---- 3. judge vs human
    n_unclear = int((df["final"] == "UNCLEAR").sum())
    cell_cols = ["dataset", "band", "model"]

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        n_valid = frame.groupby(cell_cols)["item_id"].transform("size")
        frame["weight"] = frame["cell_pop"] / n_valid
        return frame

    valid = prepare(df[df["final"].isin(["CORRECT", "INCORRECT"])])
    if valid.empty:
        sys.exit("[error] no item has a CORRECT / INCORRECT final label.")
    sampled_cells = key.groupby(cell_cols)["cell_pop"].first()
    kept_cells = valid.groupby(cell_cols)["cell_pop"].first()
    coverage = kept_cells.sum() / sampled_cells.sum()

    taus = [float(t) for t in meta["thresholds"]]
    sweep = sorted(set(np.arange(0, 100, args.sweep_step).tolist()) | set(taus))
    design = Design(valid, meta, sweep)
    h = (valid["final"] == "CORRECT").to_numpy().astype(np.int64)
    score = valid["judge_score"].to_numpy(dtype=float)
    cell = valid.groupby(cell_cols).ngroup().to_numpy()
    est = design.compute(h, score)
    lo_hi = bootstrap(design, h, score, cell, args.n_boot, rng)
    lo = {k: v[0] for k, v in lo_hi.items()}
    hi = {k: v[1] for k, v in lo_hi.items()}

    rows = []
    for t, tau in enumerate(taus):
        j = judge_correct(score, tau, design.rule)
        scopes = [(ds, f"ds.", (t, i), design.ds == i) for i, ds in enumerate(design.datasets)]
        scopes += [("ALL (pooled)", "pooled.", (t,), np.ones(len(h), bool)),
                   ("MACRO (mean of datasets)", "macro.", (t,), None)]
        for name, prefix, ix, mask in scopes:
            row = {"threshold": tau, "scope": name,
                   "n": int(mask.sum()) if mask is not None else len(h)}
            for mname in METRICS:
                k = prefix + mname
                row[mname] = float(est[k][ix])
                if mname in ("accuracy", "kappa"):
                    row[mname + "_lo"] = float(lo[k][ix])
                    row[mname + "_hi"] = float(hi[k][ix])
            if mask is not None and mask.sum():
                row["accuracy_unweighted"] = float((h[mask] == j[mask]).mean())
                row["kappa_unweighted"] = cohen_kappa(h[mask], j[mask])
            rows.append(row)
    jvh = pd.DataFrame(rows)
    jvh.to_csv(out / "judge_vs_human.csv", index=False)

    rows = []
    for t, tau in enumerate(taus):
        for i, m in enumerate(design.models):
            row = {"threshold": tau, "model": m, "n": int((design.md == i).sum())}
            for mname in ("accuracy", "kappa", "false_pos_rate", "recall"):
                k = f"model.{mname}"
                row[mname] = float(est[k][t, i])
                row[mname + "_lo"], row[mname + "_hi"] = float(lo[k][t, i]), float(hi[k][t, i])
            rows.append(row)
    by_model = pd.DataFrame(rows)
    by_model.to_csv(out / "judge_vs_human_by_model.csv", index=False)

    rows = []
    for i, ds in enumerate(design.datasets + ["MACRO (mean of datasets)"]):
        for bi, bl in enumerate(design.bands):
            if i < len(design.datasets):
                n = int(((design.ds == i) & (design.band == bi)).sum())
                e, l, u = est["band.rate"][i, bi], lo["band.rate"][i, bi], hi["band.rate"][i, bi]
            else:
                n = int((design.band == bi).sum())
                e, l, u = (est["band.macro_rate"][bi], lo["band.macro_rate"][bi],
                           hi["band.macro_rate"][bi])
            rows.append({"dataset": ds, "judge_score_band": bl, "n": n,
                         "human_correct_rate": float(e), "lo": float(l), "hi": float(u)})
    bands = pd.DataFrame(rows)
    bands.to_csv(out / "human_correct_rate_by_band.csv", index=False)

    sw = pd.DataFrame({"threshold": sweep,
                       "macro_kappa": est["sweep.macro_kappa"],
                       "macro_kappa_lo": lo["sweep.macro_kappa"],
                       "macro_kappa_hi": hi["sweep.macro_kappa"],
                       "pooled_kappa": est["sweep.pooled_kappa"],
                       "pooled_accuracy": est["sweep.pooled_accuracy"]})
    for i, ds in enumerate(design.datasets):
        sw[f"kappa_{ds}"] = est["sweep.ds_kappa"][:, i]
    sw.to_csv(out / "threshold_sweep.csv", index=False)

    tag_any = (valid["issue_tag_A"].astype(str) + "|" + valid["issue_tag_B"].astype(str))
    tag_rows = []
    for tags, bl in zip(tag_any, valid["band_label"]):
        for tg in {x.strip().upper() for x in tags.split("|") if x.strip()}:
            tag_rows.append({"issue_tag": tg, "band": bl})
    tags_tbl = (pd.DataFrame(tag_rows).groupby(["issue_tag", "band"]).size().unstack("band")
                .reindex(columns=design.bands).fillna(0).astype(int).reset_index()
                if tag_rows else pd.DataFrame())
    if len(tags_tbl):
        tags_tbl.to_csv(out / "issue_tags_by_band.csv", index=False)

    # sensitivity: UNCLEAR counted as INCORRECT
    sens = None
    if n_unclear:
        allv = prepare(df)
        d2 = Design(allv, meta, sweep)
        sens = d2.compute((allv["final"] == "CORRECT").to_numpy().astype(np.int64),
                          allv["judge_score"].to_numpy(dtype=float))

    # ---- 4. summary
    t_lo, t_hi = f"{taus[0]:g}", f"{taus[-1]:g}"
    dk, dk_lo, dk_hi = (float(est["diff.macro_kappa"]), float(lo["diff.macro_kappa"]),
                        float(hi["diff.macro_kappa"]))
    if not np.isfinite(dk):
        verdict = "The difference could not be estimated."
    elif dk_lo > 0:
        verdict = (f"The judge agrees with the human labels better at threshold {t_hi} "
                   "(interval excludes 0).")
    elif dk_hi < 0:
        verdict = (f"The judge agrees with the human labels better at threshold {t_lo} "
                   "(interval excludes 0).")
    else:
        verdict = ("The two thresholds are not distinguishable on this sample "
                   "(interval includes 0).")
    best = sw.loc[sw["macro_kappa"].idxmax()]

    s = ["# Human validation of the correctness judge", ""]
    s += [f"- main sample: {meta['n_main']} responses; with a final label: {len(df)}; "
          f"CORRECT/INCORRECT: {len(valid)}; UNCLEAR: {n_unclear}",
          f"- double-annotated: {len(dbl)}; disagreements adjudicated: {len(dis)}",
          f"- correctness rule: judge score {design.rule} threshold; "
          f"thresholds: {', '.join(f'{t:g}' for t in taus)}",
          f"- population share covered by cells with at least one valid label: {coverage:.1%}",
          f"- intervals: 95% stratified bootstrap, {args.n_boot} resamples, seed {args.seed}",
          "- all estimates are weighted to the population of judged responses unless "
          "marked unweighted", ""]
    s += ["## 1. Inter-annotator agreement", "", md_table(iaa.fillna("")), ""]
    s += ["## 2. Judge vs human label", ""]
    show = jvh.copy()
    show["accuracy"] = [with_ci(r.accuracy, r.accuracy_lo, r.accuracy_hi)
                        for r in jvh.itertuples()]
    show["kappa"] = [with_ci(r.kappa, r.kappa_lo, r.kappa_hi) for r in jvh.itertuples()]
    show["threshold"] = show["threshold"].map(lambda v: f"{v:g}")
    s += [md_table(show[["threshold", "scope", "n", "human_correct_rate", "judge_correct_rate",
                         "accuracy", "kappa", "precision", "recall", "false_pos_rate"]]), ""]
    s += ["precision = P(human CORRECT | judge correct); recall = P(judge correct | human "
          "CORRECT); false_pos_rate = P(judge correct | human INCORRECT).", ""]
    s += [f"## 3. Which threshold is closer to the human labels?", "",
          f"- macro kappa at {t_lo}: "
          f"{with_ci(est['macro.kappa'][0], lo['macro.kappa'][0], hi['macro.kappa'][0])}",
          f"- macro kappa at {t_hi}: "
          f"{with_ci(est['macro.kappa'][-1], lo['macro.kappa'][-1], hi['macro.kappa'][-1])}",
          f"- difference ({t_hi} minus {t_lo}): {with_ci(dk, dk_lo, dk_hi)}",
          f"- {verdict}",
          "- per dataset, difference in kappa: " + "; ".join(
              f"{ds} {with_ci(est['diff.ds_kappa'][i], lo['diff.ds_kappa'][i], hi['diff.ds_kappa'][i])}"
              for i, ds in enumerate(design.datasets)),
          f"- threshold with the highest macro kappa in the sweep: {best['threshold']:g} "
          f"(kappa {best['macro_kappa']:.3f}); see threshold_sweep.csv", ""]
    s += ["## 4. Share of human-CORRECT responses inside each judge-score band", ""]
    showb = bands.copy()
    showb["human_correct_rate"] = [with_ci(r.human_correct_rate, r.lo, r.hi)
                                   for r in bands.itertuples()]
    s += [md_table(showb[["dataset", "judge_score_band", "n", "human_correct_rate"]]), ""]
    s += ["## 5. By generator", ""]
    showm = by_model.copy()
    for mname in ("accuracy", "kappa", "false_pos_rate", "recall"):
        showm[mname] = [with_ci(getattr(r, mname), getattr(r, mname + "_lo"),
                                getattr(r, mname + "_hi")) for r in by_model.itertuples()]
    showm["threshold"] = showm["threshold"].map(lambda v: f"{v:g}")
    s += [md_table(showm[["threshold", "model", "n", "accuracy", "kappa", "false_pos_rate",
                          "recall"]]), ""]
    if design.has_family:
        s += [f"Same-family check: generators {meta['judge_family_models']} minus the others "
              "(a positive false_pos_rate difference means the judge is more lenient "
              "towards its own family):", ""]
        for t, tau in enumerate(taus):
            parts = [f"{mname} {with_ci(est[f'family_diff.{mname}'][t], lo[f'family_diff.{mname}'][t], hi[f'family_diff.{mname}'][t])}"
                     for mname in ("false_pos_rate", "accuracy", "kappa")]
            s.append(f"- threshold {tau:g}: " + "; ".join(parts))
        s.append("")
    if sens is not None:
        s += ["## 6. Sensitivity: UNCLEAR counted as INCORRECT", ""]
        for t, tau in enumerate(taus):
            s.append(f"- threshold {tau:g}: macro kappa {f3(sens['macro.kappa'][t])}, "
                     f"pooled accuracy {f3(sens['pooled.accuracy'][t])}")
        s.append("")
    if len(tags_tbl):
        s += ["## Issue tags by judge-score band (items tagged by either annotator)", "",
              md_table(tags_tbl), ""]
    (out / "summary.md").write_text("\n".join(s), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.fill_between(sw["threshold"], sw["macro_kappa_lo"], sw["macro_kappa_hi"],
                        color="0.8", label="95% CI (macro)")
        for ds in design.datasets:
            ax.plot(sw["threshold"], sw[f"kappa_{ds}"], lw=1, label=ds)
        ax.plot(sw["threshold"], sw["macro_kappa"], color="black", lw=2, label="macro")
        for tau in taus:
            ax.axvline(tau, color="0.4", ls=":", lw=1)
        ax.set_xlabel("judge-score threshold")
        ax.set_ylabel("Cohen's kappa, judge vs human")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "threshold_sweep.png", dpi=200)
    except Exception as exc:  # plotting is optional
        print(f"[info] plot skipped ({exc})")

    print("\n" + "\n".join(s[s.index("## 2. Judge vs human label"):]))
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
