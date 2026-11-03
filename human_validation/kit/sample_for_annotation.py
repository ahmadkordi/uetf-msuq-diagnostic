#!/usr/bin/env python3
"""Stratified sampling of judged responses for blind human annotation.

Typical use
-----------
  python sample_for_annotation.py --config config.yaml --inspect   # look at your files
  python sample_for_annotation.py --config config.yaml --dry-run   # population counts only
  python sample_for_annotation.py --config config.yaml             # write the sheets

Design
------
Unit      : one response (dataset, model, query, sample index).
Strata    : dataset x judge-score band (bands are cut at the correctness thresholds).
Per stratum: `per_stratum` responses, spread evenly over the generators, with at most
            one response per query (and, where possible, per group such as a CoQA
            conversation).
Blinding  : sheets contain no model name and no judge score; the link is in
            KEY_do_not_open.csv, which annotators must not see.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from annot_common import ISSUE_TAGS, LABELS, TEXT_COLUMNS, clean_text, have_openpyxl, write_sheet

SCALAR_FIELDS = ["question", "reference", "context", "other_references", "incorrect_references"]
KEY_NAME = "KEY_do_not_open.csv"

SHEET_INSTRUCTIONS = [
    "Read the annotation guidelines before you start. Short reminder:",
    "",
    "label = CORRECT   : the response answers the question and means the same as the "
    "reference answer (or one of the other acceptable answers).",
    "label = INCORRECT : wrong, contradicts the reference, no answer / refusal, or it does "
    "not commit to one answer.",
    "label = UNCLEAR   : you cannot decide from what is shown (ambiguous question, reference "
    "looks wrong, unreadable text). Use rarely and say why in notes.",
    "",
    "Judge the meaning, not the wording. Judge only the answer to THIS question; ignore text "
    "the model kept generating afterwards unless it contradicts the answer.",
    "issue_tag is optional: fill it when the label is not obvious. notes is free text.",
    "Work alone. Do not discuss items with the other annotator until both sheets are finished.",
    "Do not sort or delete rows and do not change item_id.",
]


# --------------------------------------------------------------------------- config
def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            sys.exit("[error] PyYAML is not installed. Run `pip install pyyaml` or give a "
                     ".json config with the same keys.")
        cfg = yaml.safe_load(text)
    cfg.setdefault("seed", 20261011)
    cfg.setdefault("output_dir", "annotation_out")
    cfg.setdefault("thresholds", [70, 90])
    cfg.setdefault("correct_if", ">")
    cfg.setdefault("score_multiplier", 1.0)
    cfg.setdefault("per_stratum", 40)
    cfg.setdefault("pilot_per_stratum", 3)
    cfg.setdefault("double_annotation_fraction", 1.0)
    cfg.setdefault("group_regex", {})
    cfg.setdefault("judge_family_models", [])
    cfg.setdefault("include_query_ids_file", None)
    cfg["thresholds"] = sorted(float(t) for t in cfg["thresholds"])
    if cfg["correct_if"] not in (">", ">="):
        sys.exit("[error] correct_if must be '>' or '>='.")
    for key in ("datasets", "models", "sources"):
        if not cfg.get(key):
            sys.exit(f"[error] config key '{key}' is missing or empty.")
    return cfg


# --------------------------------------------------------------------------- loading
def get_path(obj, path):
    """Follow a dotted path ('a.b.0.c') through dicts / lists; None if absent."""
    if path is None or path == "":
        return None
    cur = obj
    for part in str(path).split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.lstrip("-").isdigit():
            idx = int(part)
            if not -len(cur) <= idx < len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def _as_records(obj, records_path):
    obj = get_path(obj, records_path) if records_path else obj
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, dict):
        recs = []
        for k, v in obj.items():
            rec = dict(v) if isinstance(v, dict) else {"__value__": v}
            rec["__key__"] = k
            recs.append(rec)
        return recs
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise ValueError(f"cannot turn object of type {type(obj).__name__} into records")


_FILE_CACHE: dict = {}


def read_records(path: Path, records_path=None) -> list:
    cache_key = (str(path), records_path)
    if cache_key in _FILE_CACHE:
        return _FILE_CACHE[cache_key]
    name = path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    stem = name[:-3] if name.endswith(".gz") else name
    if stem.endswith((".jsonl", ".ndjson")):
        with opener(path, "rt", encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
    elif stem.endswith(".json"):
        with opener(path, "rt", encoding="utf-8") as fh:
            recs = _as_records(json.load(fh), records_path)
    elif stem.endswith((".csv", ".tsv")):
        sep = "\t" if stem.endswith(".tsv") else ","
        recs = pd.read_csv(path, sep=sep).to_dict("records")
    elif stem.endswith(".parquet"):
        recs = pd.read_parquet(path).to_dict("records")
    elif stem.endswith((".pkl", ".pickle")):
        with opener(path, "rb") as fh:
            recs = _as_records(pickle.load(fh), records_path)
    else:
        raise ValueError(f"unsupported file type: {path.name}")
    _FILE_CACHE[cache_key] = recs
    return recs


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return False


def _as_list(v):
    """Lists stored as numpy arrays / dicts keyed by index -> plain python list."""
    if _is_missing(v):
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        try:
            keys = sorted(v, key=lambda k: int(k))
        except (TypeError, ValueError):
            keys = list(v)
        return [v[k] for k in keys]
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _item(v, item_key):
    if item_key and isinstance(v, dict):
        return get_path(v, item_key)
    return v


def source_tables(src: dict, dataset: str, model: str):
    """Return (response_level_df, query_level_dict) for one source and one cell."""
    fields = dict(src.get("fields") or {})
    path_tpl = src["path"]
    ov = (src.get("overrides") or {}).get(dataset) or {}
    fields.update(ov.get("fields") or {})
    path_tpl = ov.get("path", path_tpl)
    only = src.get("datasets")
    if only and dataset not in only:
        return None, {}
    path = Path(str(path_tpl).format(dataset=dataset, model=model))
    if not path.exists():
        raise FileNotFoundError(str(path))
    recs = read_records(path, src.get("records_path"))
    layout = src.get("layout", "nested")
    if layout not in ("nested", "long"):
        raise ValueError(f"source '{src.get('name')}': layout must be nested or long")
    if "query_id" not in fields:
        raise ValueError(f"source '{src.get('name')}': fields.query_id is required")

    rows, qlevel, counters = [], {}, {}
    has_resp = bool(fields.get("responses" if layout == "nested" else "response"))
    has_score = bool(fields.get("scores" if layout == "nested" else "score"))
    for rec in recs:
        qid = get_path(rec, fields["query_id"])
        if _is_missing(qid):
            continue
        qid = str(qid)
        scal = {k: get_path(rec, fields[k]) for k in SCALAR_FIELDS if fields.get(k)}
        if scal:
            slot = qlevel.setdefault(qid, {})
            for k, v in scal.items():
                if not _is_missing(v) and k not in slot:
                    slot[k] = v
        if layout == "nested":
            resp = _as_list(get_path(rec, fields["responses"])) if has_resp else None
            scores = _as_list(get_path(rec, fields["scores"])) if has_score else None
            if resp is not None and scores is not None and len(resp) != len(scores):
                raise ValueError(
                    f"{path}: query {qid}: {len(resp)} responses but {len(scores)} scores")
            n = len(resp) if resp is not None else (len(scores) if scores is not None else 0)
            for i in range(n):
                row = {"query_id": qid, "sample_idx": i}
                if resp is not None:
                    row["response"] = _item(resp[i], fields.get("response_item_key"))
                if scores is not None:
                    row["judge_score"] = _item(scores[i], fields.get("score_item_key"))
                rows.append(row)
        else:  # long
            if not (has_resp or has_score):
                continue
            if fields.get("sample_idx"):
                idx = int(get_path(rec, fields["sample_idx"]))
            else:
                idx = counters.get(qid, 0)
                counters[qid] = idx + 1
            row = {"query_id": qid, "sample_idx": idx}
            if has_resp:
                row["response"] = get_path(rec, fields["response"])
            if has_score:
                row["judge_score"] = get_path(rec, fields["score"])
            rows.append(row)
    df = pd.DataFrame(rows) if rows else None
    return df, qlevel


def _join_refs(v) -> str:
    """Reference answers may be a string, a list, or a HF-style dict."""
    if _is_missing(v):
        return ""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        for k in ("text", "input_text", "answer", "answers", "aliases"):
            if k in v:
                return _join_refs(v[k])
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        seen, out = set(), []
        for x in v:
            s = _join_refs(x).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return " | ".join(out)
    return str(v)


def load_cell(cfg: dict, dataset: str, model: str) -> pd.DataFrame:
    resp_df, qlevel = None, {}
    for src in cfg["sources"]:
        df, ql = source_tables(src, dataset, model)
        for qid, slot in ql.items():
            tgt = qlevel.setdefault(qid, {})
            for k, v in slot.items():
                tgt.setdefault(k, v)
        if df is None:
            continue
        if resp_df is None:
            resp_df = df
        else:
            shared = [c for c in ("response", "judge_score") if c in df and c in resp_df]
            df = df.drop(columns=shared)
            resp_df = resp_df.merge(df, on=["query_id", "sample_idx"], how="inner")
    if resp_df is None:
        raise ValueError(f"no response-level data for {dataset}/{model}")
    for col in ("response", "judge_score"):
        if col not in resp_df:
            raise ValueError(f"{dataset}/{model}: no source provides '{col}'. "
                             "Check the `fields` of your sources.")
    resp_df["dataset"], resp_df["model"] = dataset, model
    resp_df["judge_score"] = (pd.to_numeric(resp_df["judge_score"], errors="coerce")
                              * float(cfg["score_multiplier"]))
    for k in SCALAR_FIELDS:
        resp_df[k] = resp_df["query_id"].map(lambda q, k=k: qlevel.get(q, {}).get(k))
    return resp_df


def load_all(cfg: dict, allow_missing: bool) -> pd.DataFrame:
    frames, missing = [], []
    for ds in cfg["datasets"]:
        for m in cfg["models"]:
            try:
                frames.append(load_cell(cfg, ds, m))
            except FileNotFoundError as exc:
                missing.append(f"{ds}/{m}: {exc}")
    if missing:
        msg = "\n  ".join(missing)
        if not allow_missing or not frames:
            sys.exit(f"[error] files not found:\n  {msg}\n"
                     "Fix the `path` templates, or pass --allow-missing to skip these cells.")
        print(f"[warn] skipped cells with missing files:\n  {msg}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- inspect
def _describe(v, depth=0, max_depth=3) -> str:
    pad = "    " * (depth + 1)
    if isinstance(v, dict):
        if depth >= max_depth:
            return f"dict({len(v)} keys)"
        lines = [f"dict({len(v)} keys)"]
        for k in list(v)[:25]:
            lines.append(f"{pad}{k}: {_describe(v[k], depth + 1, max_depth)}")
        if len(v) > 25:
            lines.append(f"{pad}... {len(v) - 25} more keys")
        return "\n".join(lines)
    if isinstance(v, (list, tuple, np.ndarray)):
        n = len(v)
        first = _describe(v[0], depth + 1, max_depth) if n else "-"
        return f"list(len={n}) of: {first}"
    s = repr(v)
    return f"{type(v).__name__} = {s[:90]}{'...' if len(s) > 90 else ''}"


def inspect(cfg: dict) -> None:
    for src in cfg["sources"]:
        print(f"\n=== source '{src.get('name', '?')}'  (layout: {src.get('layout', 'nested')})")
        shown = False
        for ds in cfg["datasets"]:
            ov = (src.get("overrides") or {}).get(ds) or {}
            for m in cfg["models"]:
                path = Path(str(ov.get("path", src["path"])).format(dataset=ds, model=m))
                status = "ok" if path.exists() else "MISSING"
                print(f"  [{status:7}] {path}")
                if path.exists() and not shown:
                    recs = read_records(path, src.get("records_path"))
                    print(f"  records: {len(recs)}; first record:")
                    print("    " + _describe(recs[0]) if recs else "    (empty)")
                    shown = True
    print("\nUse the key names above to fill `fields` in the config "
          "(dotted paths such as meta.answer are allowed; '__key__' = the key of a "
          "top-level JSON dict).")


# --------------------------------------------------------------------------- sampling
def band_labels(thresholds, rule):
    t = [f"{x:g}" for x in thresholds]
    lo, hi = ("le", "gt") if rule == ">" else ("lt", "ge")
    labels = [f"{lo}{t[0]}"]
    labels += [f"{t[i]}to{t[i + 1]}" for i in range(len(t) - 1)]
    labels.append(f"{hi}{t[-1]}")
    return labels


def add_bands(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    th = np.asarray(cfg["thresholds"], dtype=float)
    side = "left" if cfg["correct_if"] == ">" else "right"
    df["band"] = np.searchsorted(th, df["judge_score"].to_numpy(), side=side)
    labels = band_labels(cfg["thresholds"], cfg["correct_if"])
    df["band_label"] = df["band"].map(lambda b: labels[b])
    return df


def add_groups(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df["group_id"] = df["query_id"]
    for ds, pattern in (cfg.get("group_regex") or {}).items():
        rx = re.compile(pattern)
        mask = df["dataset"] == ds
        if not mask.any():
            continue

        def grp(q, rx=rx):
            m = rx.match(q)
            return m.group(1) if m and m.groups() else q
        df.loc[mask, "group_id"] = df.loc[mask, "query_id"].map(grp)
    return df


def _take(cell_idx: np.ndarray, quota: int, df: pd.DataFrame, used: dict, rng,
          level: str) -> list:
    """Pick up to `quota` rows from a cell.

    level 'group': skip rows whose group (e.g. CoQA conversation) was already used.
    level 'query': only skip rows whose query was already used (relaxed fallback).
    """
    picked = []
    if quota <= 0 or len(cell_idx) == 0:
        return picked
    ds = df["dataset"].to_numpy()
    grp = df["group_id"].to_numpy()
    qid = df["query_id"].to_numpy()
    for i in rng.permutation(cell_idx):
        gkey, qkey = (ds[i], grp[i]), (ds[i], qid[i])
        if qkey in used["query"] or (level == "group" and gkey in used["group"]):
            continue
        used["query"].add(qkey)
        used["group"].add(gkey)
        picked.append(int(i))
        if len(picked) >= quota:
            break
    return picked


def quota_per_band(value, labels: list) -> dict:
    """`per_stratum` may be one number or a mapping band_label -> number."""
    if isinstance(value, dict):
        unknown = set(map(str, value)) - set(labels)
        if unknown:
            sys.exit(f"[error] per_stratum has unknown band(s) {sorted(unknown)}; "
                     f"bands are {labels}.")
        return {i: int(value.get(lab, 0)) for i, lab in enumerate(labels)}
    return {i: int(value) for i in range(len(labels))}


def draw(df: pd.DataFrame, quota: dict, cfg: dict, used: dict, rng):
    """Stratified draw: quota[band] rows per dataset x band, balanced over models.

    Returns (row indices, number of rows that needed the relaxed 'query' level).
    """
    picks, relaxed = [], 0
    sizes = df.groupby(["dataset", "band"]).size().sort_values()  # scarce strata first
    for (ds, band), _ in sizes.items():
        per_stratum = quota.get(int(band), 0)
        if per_stratum <= 0:
            continue
        sub = df.index[(df["dataset"] == ds) & (df["band"] == band)].to_numpy()
        present = set(df.loc[sub, "model"])
        models = list(rng.permutation([m for m in cfg["models"] if m in present]))
        cells = {m: sub[(df.loc[sub, "model"] == m).to_numpy()] for m in models}
        base, rem = divmod(per_stratum, len(models))
        target = {m: base + (1 if j < rem else 0) for j, m in enumerate(models)}
        have = {m: 0 for m in models}
        for level in ("group", "query"):
            new = []
            for m in models:  # balanced over generators
                got = _take(cells[m], target[m] - have[m], df, used, rng, level)
                have[m] += len(got)
                new += got
            for m in models:  # a short cell: fill the stratum from the other generators
                missing = per_stratum - sum(have.values())
                if missing <= 0:
                    break
                got = _take(cells[m], missing, df, used, rng, level)
                have[m] += len(got)
                new += got
            picks += new
            if level == "query":
                relaxed += len(new)
            if sum(have.values()) >= per_stratum:
                break
    return picks, relaxed


def make_ids(n: int, rng) -> list:
    alphabet = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"))
    ids = set()
    while len(ids) < n:
        ids.add("".join(rng.choice(alphabet, size=6)))
    out = sorted(ids)
    rng.shuffle(out)
    return out


def sheet_frame(items: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame:
    """Blind view: grouped by dataset, shuffled within dataset; drop empty text columns."""
    parts = []
    for ds in cfg["datasets"]:
        block = items[items["dataset"] == ds]
        if len(block):
            parts.append(block.iloc[rng.permutation(len(block))])
    view = pd.concat(parts, ignore_index=True) if parts else items.iloc[:0]
    cols = ["item_id", "dataset"]
    cols += [c for c in TEXT_COLUMNS
             if c in ("question", "reference_answer", "model_response")
             or view[c].astype(str).str.strip().ne("").any()]
    return view[cols]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--inspect", action="store_true",
                    help="print the structure of the first record of each source and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="load everything, print population counts, write nothing")
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip dataset/model cells whose files are missing")
    ap.add_argument("--format", choices=["xlsx", "csv"], default="xlsx")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output directory (destroys earlier sheets)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.inspect:
        inspect(cfg)
        return

    df = load_all(cfg, args.allow_missing)
    n_loaded = len(df)

    if cfg.get("include_query_ids_file"):
        keep = {ln.strip() for ln in Path(cfg["include_query_ids_file"]).read_text().splitlines()
                if ln.strip()}
        df = df[df["query_id"].isin(keep)].reset_index(drop=True)
        print(f"[info] restricted to {len(keep)} listed query ids: {len(df)} responses left")

    n_nan = int(df["judge_score"].isna().sum())
    df = df[df["judge_score"].notna()].reset_index(drop=True)
    if df.empty:
        sys.exit("[error] no response has a numeric judge score. Check `fields.scores`.")
    smax = df["judge_score"].max()
    if smax <= 1.0 + 1e-9 and max(cfg["thresholds"]) > 1:
        sys.exit(f"[error] judge scores go up to {smax:g} but thresholds are "
                 f"{cfg['thresholds']}. Set score_multiplier: 100 in the config.")
    df = add_groups(add_bands(df, cfg), cfg)
    labels = band_labels(cfg["thresholds"], cfg["correct_if"])

    pop = (df.groupby(["dataset", "model", "band_label"]).size()
             .unstack("band_label").reindex(columns=labels).fillna(0).astype(int))
    print(f"\nLoaded {n_loaded} responses; {n_nan} without a numeric judge score were dropped.")
    print(f"Queries: {df.groupby('dataset')['query_id'].nunique().to_dict()}")
    print("\nPopulation (responses) per dataset / model / judge-score band:")
    print(pop.to_string())
    acc = pd.DataFrame({
        f"correct@{t:g}": (df["judge_score"] > t if cfg["correct_if"] == ">"
                           else df["judge_score"] >= t).groupby(
                               [df["dataset"], df["model"]]).mean()
        for t in cfg["thresholds"]})
    print("\nShare of responses the judge counts as correct (compare with the mean sample "
          "accuracy in your paper):")
    print(acc.round(3).to_string())
    miss_q = float((df["question"].map(_is_missing)).mean())
    miss_r = float((df["reference"].map(_is_missing)).mean())
    if miss_q > 0 or miss_r > 0:
        print(f"\n[warn] missing question for {miss_q:.1%} and missing reference for "
              f"{miss_r:.1%} of responses - check `fields.question` / `fields.reference`.")
    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    out = Path(cfg["output_dir"])
    if (out / KEY_NAME).exists() and not args.force:
        sys.exit(f"[error] {out / KEY_NAME} already exists. Re-sampling would invalidate "
                 "labels already collected. Use a new output_dir, or --force to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg["seed"]))
    used = {"group": set(), "query": set()}
    main_quota = quota_per_band(cfg["per_stratum"], labels)
    main_idx, relaxed_main = draw(df, main_quota, cfg, used, rng)
    pilot_idx, _ = draw(df, quota_per_band(cfg["pilot_per_stratum"], labels), cfg, used, rng)

    items = df.loc[main_idx + pilot_idx].copy()
    items["set"] = ["main"] * len(main_idx) + ["pilot"] * len(pilot_idx)
    items["item_id"] = make_ids(len(items), rng)
    items["question"] = items["question"].map(clean_text)
    items["context"] = items["context"].map(clean_text)
    items["reference_answer"] = items["reference"].map(_join_refs)
    items["other_acceptable_answers"] = items["other_references"].map(_join_refs)
    items["known_incorrect_answers"] = items["incorrect_references"].map(_join_refs)

    def _drop_dupes(row):
        ref = {r.strip().lower() for r in row["reference_answer"].split(" | ")}
        keep = [o for o in row["other_acceptable_answers"].split(" | ")
                if o.strip() and o.strip().lower() not in ref]
        return " | ".join(keep)
    items["other_acceptable_answers"] = items.apply(_drop_dupes, axis=1)
    items["model_response"] = items["response"].map(
        lambda v: "[EMPTY RESPONSE]" if clean_text(v).strip() == "" else clean_text(v))

    # population and sample size of each dataset x band x model cell (for weights)
    cell = ["dataset", "band", "model"]
    pop_cell = df.groupby(cell).size().rename("cell_pop")
    n_cell = items[items["set"] == "main"].groupby(cell).size().rename("cell_n")
    items = items.join(pop_cell, on=cell).join(n_cell, on=cell)
    items["cell_n"] = items["cell_n"].fillna(0).astype(int)

    # which main items the second annotator also labels
    frac = float(cfg["double_annotation_fraction"])
    items["double_annotated"] = items["set"] == "pilot"
    for _, block in items[items["set"] == "main"].groupby(["dataset", "band"]):
        k = len(block) if frac >= 1 else int(round(frac * len(block)))
        chosen = rng.permutation(block.index.to_numpy())[:k]
        items.loc[chosen, "double_annotated"] = True

    inputs = {"label": LABELS, "issue_tag": ISSUE_TAGS, "notes": None}
    main_items = items[items["set"] == "main"]
    pilot_items = items[items["set"] == "pilot"]
    written = []
    written.append(write_sheet(out / "annotator_A", sheet_frame(main_items, cfg, rng), inputs,
                               args.format, SHEET_INSTRUCTIONS))
    written.append(write_sheet(out / "annotator_B",
                               sheet_frame(main_items[main_items["double_annotated"]], cfg, rng),
                               inputs, args.format, SHEET_INSTRUCTIONS))
    if len(pilot_items):
        for who in ("A", "B"):
            written.append(write_sheet(out / f"pilot_{who}", sheet_frame(pilot_items, cfg, rng),
                                       inputs, args.format, SHEET_INSTRUCTIONS))

    key_cols = ["item_id", "set", "dataset", "model", "query_id", "sample_idx", "judge_score",
                "band", "band_label", "cell_pop", "cell_n", "double_annotated"]
    key_path = out / KEY_NAME
    items[key_cols].to_csv(key_path, index=False)
    pop.to_csv(out / "population_counts.csv")

    meta = {
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": int(cfg["seed"]),
        "thresholds": cfg["thresholds"],
        "correct_if": cfg["correct_if"],
        "band_labels": labels,
        "datasets": list(cfg["datasets"]),
        "models": list(cfg["models"]),
        "judge_family_models": list(cfg.get("judge_family_models") or []),
        "per_stratum": {labels[i]: n for i, n in main_quota.items()},
        "pilot_per_stratum": cfg["pilot_per_stratum"],
        "double_annotation_fraction": frac,
        "n_main": int(len(main_items)),
        "n_pilot": int(len(pilot_items)),
        "n_population": int(len(df)),
        "n_dropped_no_score": n_nan,
        "n_main_sharing_a_group": int(relaxed_main),
        "config_sha256": sha256(args.config),
        "key_sha256": sha256(key_path),
    }
    (out / "sampling_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    samp = (main_items.groupby(["dataset", "band_label", "model"]).size()
            .unstack("model").fillna(0).astype(int))
    samp["total"] = samp.sum(axis=1)
    report = [
        "# Sampling report", "",
        f"- created: {meta['created']}",
        f"- seed: {meta['seed']}; thresholds: {meta['thresholds']}; "
        f"correct if score {meta['correct_if']} threshold",
        f"- population: {meta['n_population']} responses "
        f"({n_nan} dropped for a missing score)",
        f"- main sample: {meta['n_main']} responses; pilot: {meta['n_pilot']}",
        f"- double annotation fraction: {frac:g}",
        f"- main items sharing a group with another item: {relaxed_main}",
        f"- sha256 of {KEY_NAME}: `{meta['key_sha256']}`", "",
        "## Main sample per dataset x band x model", "",
        "```", samp.to_string(), "```", "",
        "## Population (responses) per dataset x model x band", "",
        "```", pop.to_string(), "```", "",
    ]
    (out / "sampling_report.md").write_text("\n".join(report), encoding="utf-8")

    wanted = {labels[i]: n for i, n in main_quota.items()}
    print(f"\nMain sample per dataset x band x model (target per row: {wanted}):")
    print(samp.to_string())
    target = samp.index.get_level_values("band_label").map(wanted).to_numpy()
    short = samp[samp["total"].to_numpy() < target]
    if len(short):
        print("\n[warn] some strata have fewer responses than requested (all usable ones "
              "were taken):")
        print(short["total"].to_string())
    if relaxed_main:
        print(f"\n[warn] {relaxed_main} main items share a group (e.g. a CoQA conversation) "
              "with another item, because there were not enough unused groups. "
              "They still come from different queries.")
    if args.format == "xlsx" and not have_openpyxl():
        print("\n[warn] openpyxl is not installed; wrote CSV sheets instead "
              "(`pip install openpyxl` for Excel sheets with dropdowns).")
    print("\nWritten:")
    for p in written + [key_path, out / "sampling_meta.json", out / "sampling_report.md",
                        out / "population_counts.csv"]:
        print(f"  {p}")
    print(f"\nGive annotators ONLY the annotator_* / pilot_* sheets. Keep {KEY_NAME} away "
          "from them until adjudication is finished.")


if __name__ == "__main__":
    main()
