#!/usr/bin/env python3
"""Auto-detect all U-ETF progress files in /mnt/output and show a live table."""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path('/mnt/output')

# Expected totals per dataset
DATASET_TOTAL = {
    'coqa': 7983,
    'nqopen': 3610,
    'hotpotqa': 7405,
    'truthfulqa': 817,
}

# Stage regex → (stage_name, group order)
STAGE_PATTERNS = [
    ('A',      re.compile(r'^stage_a_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>[^_]+(?:[._][^_]+)*?)(?:_shard\d+)?(?:_merged)?\.jsonl$')),
    ('S1',     re.compile(r'^stage_b_s1_semantic_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>.+?)(?:_shard\d+)?\.jsonl$')),
    ('claims', re.compile(r'^stage_b_claims_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>.+?)(?:_shard\d+)?\.jsonl$')),
    ('S2',     re.compile(r'^stage_b_s2_claim_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>.+?)(?:_shard\d+)?\.jsonl$')),
    ('S3',     re.compile(r'^stage_b_s3_self_eval_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>.+?)(?:_shard\d+)?\.jsonl$')),
    ('C',      re.compile(r'^stage_c_tensor_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>.+?)(?:_shard\d+)?\.jsonl$')),
    ('F',      re.compile(r'^stage_f_estimators_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>.+?)(?:_shard\d+)?\.jsonl$')),
    ('G',      re.compile(r'^stage_g_calibrated_(?P<ds>coqa|nqopen|hotpotqa|truthfulqa)_(?P<model>.+?)(?:_shard\d+)?\.jsonl$')),
]

MODEL_ALIASES = {
    'Llama2_13b':    'Llama2-13B',
    'Q2.5_14b':      'Qwen2.5-14B',
    'llama3.1_8b':   'Llama3.1-8B',
}

STAGE_ORDER = ['A', 'S1', 'claims', 'S2', 'S3', 'C', 'F', 'G']


def count_lines(p):
    n = 0
    try:
        with p.open('rb') as f:
            for _ in f:
                n += 1
    except Exception:
        return 0
    return n


def scan():
    """Return dict: (ds, model, stage) -> {shards, merged_lines, total_lines}"""
    data = {}
    for f in BASE.glob('stage_*.jsonl'):
        # Skip merged files - we aggregate shards
        is_merged = '_merged' in f.name
        is_shard = bool(re.search(r'_shard\d+\.jsonl$', f.name))

        for stage, pat in STAGE_PATTERNS:
            m = pat.match(f.name)
            if not m:
                continue
            ds = m.group('ds')
            raw_model = m.group('model')
            # Normalize model name
            model = MODEL_ALIASES.get(raw_model, raw_model)

            key = (ds, model, stage)
            if key not in data:
                data[key] = {'shard_lines': 0, 'merged_lines': 0, 'shard_files': 0}

            n = count_lines(f)
            if is_shard:
                data[key]['shard_lines'] += n
                data[key]['shard_files'] += 1
            elif is_merged:
                # Ignore merged, use shard sum
                pass
            else:
                # Non-shard file (either merged output or single-shard output)
                data[key]['merged_lines'] = max(data[key]['merged_lines'], n)
            break
    return data


def bar(frac, w=6):
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * w))
    return '█' * fill + '░' * (w - fill)


def main():
    prev = 0
    first = True
    while True:
        data = scan()

        # Build display grid
        lines = []
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f"  U-ETF Auto-Monitor  |  {now}")
        lines.append("  " + "─" * 130)
        header = f"  {'Cell':<28}"
        for s in STAGE_ORDER:
            header += f" {s:>11}"
        lines.append(header)
        lines.append("  " + "─" * 130)

        datasets = ['coqa', 'nqopen', 'hotpotqa', 'truthfulqa']
        models = ['Llama2-13B', 'Qwen2.5-14B', 'Llama3.1-8B']

        for ds in datasets:
            for model in models:
                total = DATASET_TOTAL[ds]
                row = f"  {ds:<8}×{model:<17}"
                for stage in STAGE_ORDER:
                    key = (ds, model, stage)
                    if key not in data:
                        row += f" {'--':>11}"
                        continue

                    d = data[key]
                    # Prefer shard sum if any shards exist
                    n = d['shard_lines'] if d['shard_lines'] > 0 else d['merged_lines']

                    if n <= 0:
                        row += f" {'--':>11}"
                    elif n >= total:
                        row += f" {'✅ DONE':>11}"
                    else:
                        frac = n / total
                        row += f" {bar(frac, 5)} {100*frac:>3.0f}%"
                lines.append(row)
            lines.append("")

        lines.append("  " + "─" * 130)
        lines.append(f"  Total cells: {len(data)}  |  Files in /mnt/output: {len(list(BASE.glob('*.jsonl')))}")

        if first:
            for _ in lines:
                sys.stdout.write("\n")
            first = False
        else:
            sys.stdout.write(f"\033[{prev}A")

        for ln in lines:
            sys.stdout.write("\033[2K" + ln + "\n")
        sys.stdout.flush()
        prev = len(lines)

        time.sleep(15)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\n  stopped\n")
