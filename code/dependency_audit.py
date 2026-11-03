#!/usr/bin/env python3
"""Experiment 0A: Estimator/Source Dependency Audit (methodology v9, section 16.1).

Runs BEFORE any gate training and BEFORE any correctness-derived analysis.
It never reads gold answers or correctness labels -- only Stage-G estimator
values -- so it cannot leak evaluation information.

What it does, per (dataset x generator) cell:
  1. Split queries 50/20/30 (train/val/test) deterministically by hashing the
     query id (or its conversation id for CoQA), so the split is identical
     across generators, reproducible without a saved file, and independent of
     which ids happen to be present. The TEST split is never touched here.
  2. Re-fit the e-CDF on TRAIN ONLY (from the raw estimator values stored in
     the Stage-G file) and apply it to train+val -- this deliberately ignores
     the same-set (train == eval) calibration in the existing Stage-G output.
  3. Compute Spearman and Pearson correlation between every pair of
     estimators on the train+val analysis set.
  4. Report within-family and between-family dependence, with families
     defined by computational lineage:
         semantic/NLI (s1_semantic.*), claim/NLI (s2_claim.*),
         tensor-derived (tensor.*), self-evaluation (s3_self_eval.*).
  5. Mandatory leave-one-source-out (LOSO): remove one source at a time --
     INCLUDING the tensor estimators when removing S1 or S2, because the
     tensor is built from those matrices -- and measure how much of the
     apparent cross-estimator agreement disappears. A large drop means the
     agreement was carried by shared ancestry, not independent evidence.
  6. Compare simple leave-one-estimator-out consensus against the
     group-aware (family-averaged) consensus. The pool is 12 graph
     estimators + 2 tensor + 1 self-eval, so the simple consensus is ~80%
     graph-estimator-driven by construction; this quantifies how different
     the group-aware version really is.
  7. Flag highly redundant estimator clusters (|Spearman| >= threshold) and
     SUGGEST a main pool vs ablation-only pool. The suggestion is advisory --
     freezing the pool is a human decision (methodology Phase 3).

Limitations, stated plainly:
  - Orientation ("larger = more uncertain") cannot be verified against
    correctness here (that needs labels, i.e. Experiment 0B). This script
    only flags estimators that are NEGATIVELY correlated with the rest of
    the pool, which would suggest a flipped orientation.
  - The leave-one-estimator-out agreement reported here is a rank-based
    diagnostic (Spearman between an estimator and the mean of the others),
    not the exact a_k(q) = exp(-d_cons/sigma) of section 11.1.
  - Entity-wise grouping is applied to CoQA (all turns of a conversation stay
    in one split). NQ-Open / HotpotQA / TruthfulQA are split per question;
    entity/topic grouping beyond that is not attempted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.dependency_audit")

FAMILY_RULES = [
    ("s1_semantic.", "semantic_nli"),
    ("s2_claim.", "claim_nli"),
    ("tensor.", "tensor"),
    ("s3_self_eval.", "self_eval"),
]
FAMILY_ORDER = ["semantic_nli", "claim_nli", "tensor", "self_eval"]

# Removing a source also removes its descendants: the CP/Tucker tensor is
# built from S1 and S2, so it inherits their information.
LOSO_SOURCES = {
    "remove_S1_semantic(+tensor)": ("semantic_nli", "tensor"),
    "remove_S2_claim(+tensor)": ("claim_nli", "tensor"),
    "remove_S3_self_eval": ("self_eval",),
    "remove_tensor_only": ("tensor",),
}

SPLIT_FRACTIONS = (0.5, 0.2, 0.3)  # train / val / test, methodology 15.2


# --------------------------------------------------------------------------
# Loading and splitting
# --------------------------------------------------------------------------
def family_of(key: str) -> str:
    for prefix, family in FAMILY_RULES:
        if key.startswith(prefix):
            return family
    raise ValueError(f"Estimator key {key!r} does not match any known lineage family.")


def canonical_key_order(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda k: (FAMILY_ORDER.index(family_of(k)), k))


def load_stage_g(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    ids: list[str] = []
    columns: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(str(rec["id"]))
            for key, value in rec["raw"].items():
                columns.setdefault(key, []).append(float(value))
    n = len(ids)
    out: dict[str, np.ndarray] = {}
    for key, col in columns.items():
        if len(col) != n:
            raise ValueError(f"{path}: estimator {key} present in {len(col)}/{n} records.")
        out[key] = np.asarray(col, dtype=np.float64)
    return ids, out


def group_key(query_id: str, mode: str) -> str:
    if mode == "coqa":
        return re.sub(r"_\d+$", "", query_id)  # <conversation>_<turn> -> <conversation>
    return query_id


def split_of(key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < SPLIT_FRACTIONS[0]:
        return "train"
    if u < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]:
        return "val"
    return "test"


def cell_name(path: Path) -> str:
    name = path.name
    name = re.sub(r"^stage_g_calibrated_", "", name)
    return re.sub(r"\.jsonl$", "", name)


def resolve_group_mode(mode: str, cell: str) -> str:
    if mode != "auto":
        return mode
    return "coqa" if "coqa" in cell.lower() else "none"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def avg_rank(a: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    return (cum - (counts - 1) / 2.0)[inv]


def pearson_matrix(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - mean) / std
    corr = (z.T @ z) / x.shape[0]
    corr[:, std == 0] = np.nan
    corr[std == 0, :] = np.nan
    return corr


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    ranks = np.column_stack([avg_rank(x[:, j]) for j in range(x.shape[1])])
    return pearson_matrix(ranks)


def spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = avg_rank(a), avg_rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ecdf_apply(train_values: np.ndarray, x: np.ndarray) -> np.ndarray:
    sorted_train = np.sort(train_values)
    return np.searchsorted(sorted_train, x, side="right") / float(len(sorted_train))


def _nanmean(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else None


def family_blocks(corr: np.ndarray, keys: list[str]) -> dict[str, Any]:
    fam = [family_of(k) for k in keys]
    within: dict[str, Any] = {}
    between: dict[str, Any] = {}
    for f in FAMILY_ORDER:
        idx = [i for i, ff in enumerate(fam) if ff == f]
        vals = [corr[i, j] for a, i in enumerate(idx) for j in idx[a + 1:]]
        within[f] = {
            "n_estimators": len(idx),
            "mean_rho": _nanmean(vals),
            "mean_abs_rho": _nanmean([abs(v) for v in vals if np.isfinite(v)]),
        }
    for a, f in enumerate(FAMILY_ORDER):
        for g in FAMILY_ORDER[a + 1:]:
            fi = [i for i, ff in enumerate(fam) if ff == f]
            gi = [i for i, ff in enumerate(fam) if ff == g]
            vals = [corr[i, j] for i in fi for j in gi]
            between[f"{f}__{g}"] = {
                "mean_rho": _nanmean(vals),
                "mean_abs_rho": _nanmean([abs(v) for v in vals if np.isfinite(v)]),
            }
    return {"within": within, "between": between}


def loo_agreement(x: np.ndarray, cols: list[int]) -> dict[str, Any]:
    """Spearman between each estimator and the mean of the OTHER estimators
    in `cols` (leave-one-estimator-out consensus)."""
    if len(cols) < 2:
        return {"mean": None, "per_estimator": {}}
    per: dict[int, float] = {}
    for k in cols:
        others = [j for j in cols if j != k]
        per[k] = spearman_1d(x[:, k], x[:, others].mean(axis=1))
    return {"mean": _nanmean(list(per.values())), "per": per}


def mean_pairwise_abs(corr: np.ndarray, cols: list[int]) -> float | None:
    vals = [abs(corr[i, j]) for a, i in enumerate(cols) for j in cols[a + 1:] if np.isfinite(corr[i, j])]
    return _nanmean(vals)


def redundancy_clusters(corr: np.ndarray, keys: list[str], threshold: float) -> list[list[str]]:
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(find(i), []).append(k)
    return [g for g in groups.values() if len(g) > 1]


def suggest_pools(corr: np.ndarray, keys: list[str], clusters: list[list[str]]) -> dict[str, list[str]]:
    """Within each redundant cluster keep the medoid (highest mean |rho| to the
    other members); the rest become ablation-only. Advisory only."""
    index = {k: i for i, k in enumerate(keys)}
    ablation: list[str] = []
    for cluster in clusters:
        best, best_score = None, -1.0
        for k in cluster:
            others = [index[o] for o in cluster if o != k]
            score = _nanmean([abs(corr[index[k], j]) for j in others]) or 0.0
            if score > best_score:
                best, best_score = k, score
        ablation.extend(k for k in cluster if k != best)
    main = [k for k in keys if k not in ablation]
    return {"main_pool": main, "ablation_only": sorted(ablation)}


# --------------------------------------------------------------------------
# Per-cell audit
# --------------------------------------------------------------------------
def audit_cell(
    path: Path, *, seed: int, group_mode: str, threshold: float, split_dir: Path | None,
) -> dict[str, Any]:
    cell = cell_name(path)
    mode = resolve_group_mode(group_mode, cell)
    ids, raw = load_stage_g(path)
    keys_all = canonical_key_order(list(raw))

    finite = np.all(np.column_stack([np.isfinite(raw[k]) for k in keys_all]), axis=1)
    dropped_nonfinite = int((~finite).sum())
    splits = np.array([split_of(group_key(i, mode), seed) for i in ids])
    counts = {s: int(((splits == s) & finite).sum()) for s in ("train", "val", "test")}

    if split_dir is not None:
        split_dir.mkdir(parents=True, exist_ok=True)
        for s in ("train", "val", "test"):
            sel = [ids[i] for i in range(len(ids)) if splits[i] == s and finite[i]]
            (split_dir / f"{cell}.{s}_ids.txt").write_text("\n".join(sel) + "\n", encoding="utf-8")

    train_mask = (splits == "train") & finite
    analysis_mask = ((splits == "train") | (splits == "val")) & finite  # TEST is never used
    if train_mask.sum() < 30 or analysis_mask.sum() < 60:
        raise ValueError(f"{cell}: too few queries after split (train={train_mask.sum()}, analysis={analysis_mask.sum()}).")

    # Degenerate / constant estimators (on the training split, where the e-CDF is fit).
    active: list[str] = []
    degenerate: dict[str, dict[str, Any]] = {}
    for k in keys_all:
        tr = raw[k][train_mask]
        uniq, cnt = np.unique(tr, return_counts=True)
        modal = float(cnt.max() / cnt.sum())
        if uniq.size <= 1:
            degenerate[k] = {"reason": "constant_on_train", "modal_fraction": modal}
        else:
            active.append(k)
            if modal >= 0.5:
                degenerate[k] = {"reason": "heavy_ties(modal_fraction>=0.5)", "modal_fraction": modal}

    x = np.column_stack([ecdf_apply(raw[k][train_mask], raw[k][analysis_mask]) for k in active])
    spearman = spearman_matrix(x)
    pearson = pearson_matrix(x)

    fam_spear = family_blocks(spearman, active)
    fam_pear = family_blocks(pearson, active)

    all_cols = list(range(len(active)))
    full_loo = loo_agreement(x, all_cols)
    full_pairwise = mean_pairwise_abs(spearman, all_cols)

    loso: dict[str, Any] = {}
    for name, removed in LOSO_SOURCES.items():
        keep = [i for i, k in enumerate(active) if family_of(k) not in removed]
        if len(keep) < 2:
            loso[name] = {"remaining_estimators": len(keep), "loo_agreement": None, "delta_vs_full": None}
            continue
        sub = loo_agreement(x, keep)
        loso[name] = {
            "remaining_estimators": len(keep),
            "loo_agreement": sub["mean"],
            "mean_pairwise_abs_rho": mean_pairwise_abs(spearman, keep),
            "delta_vs_full": (sub["mean"] - full_loo["mean"]) if (sub["mean"] is not None and full_loo["mean"] is not None) else None,
        }

    # Simple vs group-aware (family-averaged) consensus.
    fam_cols = {f: [i for i, k in enumerate(active) if family_of(k) == f] for f in FAMILY_ORDER}
    fam_means = np.column_stack([x[:, c].mean(axis=1) for f, c in fam_cols.items() if c])
    simple_vs_group = spearman_1d(x.mean(axis=1), fam_means.mean(axis=1))
    family_share = {f: len(c) / len(active) for f, c in fam_cols.items()}

    # Orientation red flag: estimators negatively correlated with the rest of the pool.
    negative = [
        active[k] for k, v in full_loo["per"].items() if np.isfinite(v) and v < 0
    ]

    clusters = redundancy_clusters(spearman, active, threshold)
    pools = suggest_pools(spearman, active, clusters)

    if split_dir is not None:
        for name, mat in (("spearman", spearman), ("pearson", pearson)):
            with (split_dir.parent / f"{cell}.{name}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([""] + active)
                for k, row in zip(active, mat):
                    writer.writerow([k] + [f"{v:.6f}" if np.isfinite(v) else "nan" for v in row])

    return {
        "cell": cell,
        "n_records": len(ids),
        "dropped_nonfinite": dropped_nonfinite,
        "group_mode": mode,
        "split_counts": counts,
        "analysis_n": int(analysis_mask.sum()),
        "estimators_active": active,
        "estimators_degenerate": degenerate,
        "family_correlation_spearman": fam_spear,
        "family_correlation_pearson": fam_pear,
        "loo_consensus_agreement_full_pool": full_loo["mean"],
        "mean_pairwise_abs_spearman_full_pool": full_pairwise,
        "loso": loso,
        "simple_vs_group_aware_consensus_spearman": simple_vs_group,
        "family_share_of_pool": family_share,
        "negatively_correlated_with_pool": negative,
        "redundancy_threshold": threshold,
        "redundant_clusters": clusters,
        "suggested_pools": pools,
    }


# --------------------------------------------------------------------------
# Aggregation and reporting
# --------------------------------------------------------------------------
def aggregate(cells: list[dict[str, Any]], majority: float) -> dict[str, Any]:
    n = len(cells)
    ablation_votes: dict[str, int] = {}
    active_counts: dict[str, int] = {}
    for c in cells:
        for k in c["estimators_active"]:
            active_counts[k] = active_counts.get(k, 0) + 1
        for k in c["suggested_pools"]["ablation_only"]:
            ablation_votes[k] = ablation_votes.get(k, 0) + 1
    stable_ablation = sorted(k for k, v in ablation_votes.items() if v / n >= majority)

    def mean_field(getter) -> float | None:
        return _nanmean([getter(c) for c in cells])

    return {
        "n_cells": n,
        "mean_loo_consensus_agreement": mean_field(lambda c: c["loo_consensus_agreement_full_pool"]),
        "mean_simple_vs_group_consensus_spearman": mean_field(lambda c: c["simple_vs_group_aware_consensus_spearman"]),
        "mean_loso_delta": {
            name: mean_field(lambda c, name=name: c["loso"][name]["delta_vs_full"]) for name in LOSO_SOURCES
        },
        "mean_within_family_abs_rho": {
            f: mean_field(lambda c, f=f: c["family_correlation_spearman"]["within"][f]["mean_abs_rho"]) for f in FAMILY_ORDER
        },
        "ablation_votes": dict(sorted(ablation_votes.items(), key=lambda kv: -kv[1])),
        "ablation_majority_threshold": majority,
        "suggested_stable_ablation_only": stable_ablation,
        "estimator_active_in_cells": active_counts,
    }


def fmt(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def write_markdown(cells: list[dict[str, Any]], agg: dict[str, Any], path: Path) -> None:
    lines = ["# Experiment 0A -- Dependency Audit Summary", ""]
    lines += [
        f"Cells audited: **{agg['n_cells']}**. Analysis set = train+val (test never touched); "
        "e-CDF re-fit on train only.",
        "",
        "## Aggregate",
        f"- Mean leave-one-estimator-out consensus agreement (Spearman): **{fmt(agg['mean_loo_consensus_agreement'])}**",
        f"- Simple vs group-aware consensus (Spearman): **{fmt(agg['mean_simple_vs_group_consensus_spearman'])}**",
        "- Mean LOSO change in agreement (negative = agreement was carried by the removed source):",
    ]
    for name, v in agg["mean_loso_delta"].items():
        lines.append(f"  - {name}: {fmt(v)}")
    lines += ["- Mean within-family |Spearman|:"]
    for f, v in agg["mean_within_family_abs_rho"].items():
        lines.append(f"  - {f}: {fmt(v)}")
    lines += [
        "",
        f"Estimators suggested for the ablation-only pool in >= {agg['ablation_majority_threshold']:.0%} of cells "
        "(advisory -- freezing the pool is a human decision):",
    ]
    lines += [f"- `{k}`" for k in agg["suggested_stable_ablation_only"]] or ["- (none)"]
    lines += ["", "## Per cell", "", "| cell | analysis n | LOO agree | simple-vs-group | LOSO S1(+T) | LOSO S2(+T) | LOSO S3 | degenerate | neg. orientation |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lo = c["loso"]
        lines.append(
            f"| {c['cell']} | {c['analysis_n']} | {fmt(c['loo_consensus_agreement_full_pool'])} | "
            f"{fmt(c['simple_vs_group_aware_consensus_spearman'])} | "
            f"{fmt(lo['remove_S1_semantic(+tensor)']['delta_vs_full'])} | "
            f"{fmt(lo['remove_S2_claim(+tensor)']['delta_vs_full'])} | "
            f"{fmt(lo['remove_S3_self_eval']['delta_vs_full'])} | "
            f"{len(c['estimators_degenerate'])} | {len(c['negatively_correlated_with_pool'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-g", required=True, type=Path, nargs="+", help="Stage-G files (shell globs are fine)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260903, help="Split seed (must stay fixed across ALL later experiments)")
    parser.add_argument("--group-mode", choices=["auto", "coqa", "none"], default="auto",
                        help="auto = group by conversation for cells whose name contains 'coqa'")
    parser.add_argument("--redundancy-threshold", type=float, default=0.95)
    parser.add_argument("--ablation-majority", type=float, default=0.5,
                        help="Fraction of cells that must flag an estimator before it is listed as stable ablation-only")
    parser.add_argument("--min-queries", type=int, default=200,
                        help="Skip files with fewer records (e.g. the 25-query pilot files sharing the same directory)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    for path in args.stage_g:
        n_lines = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
        if n_lines < args.min_queries:
            LOGGER.info("Skipping %s (%d records < --min-queries=%d).", path.name, n_lines, args.min_queries)
            continue
        LOGGER.info("Auditing %s (%d records)...", path.name, n_lines)
        result = audit_cell(
            path, seed=args.seed, group_mode=args.group_mode,
            threshold=args.redundancy_threshold, split_dir=args.output_dir / "splits",
        )
        cells.append(result)
        LOGGER.info(
            "  %s: LOO agreement=%s | simple-vs-group=%s | degenerate=%d | clusters=%d",
            result["cell"], fmt(result["loo_consensus_agreement_full_pool"]),
            fmt(result["simple_vs_group_aware_consensus_spearman"]),
            len(result["estimators_degenerate"]), len(result["redundant_clusters"]),
        )

    if not cells:
        raise SystemExit("No cells audited (all files below --min-queries?).")

    agg = aggregate(cells, args.ablation_majority)
    (args.output_dir / "dependency_audit_report.json").write_text(
        json.dumps({"aggregate": agg, "cells": cells}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(cells, agg, args.output_dir / "dependency_audit_summary.md")
    LOGGER.info("Wrote report to %s", args.output_dir)


if __name__ == "__main__":
    main()
