#!/usr/bin/env python3
"""Monitor all U-ETF cells across all stages (updated naming)."""
import sys
import time
from datetime import datetime
from pathlib import Path

DATASETS = {'coqa': 7983, 'nqopen': 3610, 'hotpotqa': 7405, 'truthfulqa': 817}
MODELS = ['Llama2_13b', 'Q2.5_14b', 'llama3.1_8b']

STAGES = [
    ('S1',   'stage_b_s1_semantic_{ds}_{model}.jsonl'),
    ('claims','stage_b_claims_{ds}_{model}.jsonl'),
    ('S2',   'stage_b_s2_claim_{ds}_{model}.jsonl'),
    ('S3',   'stage_b_s3_self_eval_{ds}_{model}.jsonl'),
    ('C',    'stage_c_tensor_{ds}_{model}.jsonl'),
    ('F',    'stage_f_estimators_{ds}_{model}.jsonl'),
    ('G',    'stage_g_calibrated_{ds}_{model}.jsonl'),
]

BASE = Path('/mnt/output')


def count_lines(p: Path):
    if not p.exists():
        return None
    n = 0
    with p.open('rb') as f:
        for _ in f:
            n += 1
    return n


def bar(frac, w=8):
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * w))
    return '█' * fill + '░' * (w - fill)


def main():
    prev_lines = 0
    first = True
    try:
        while True:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            lines = []
            lines.append(f"  U-ETF Progress Monitor  |  {now}")
            lines.append("  " + "─" * 110)
            header = f"  {'Cell':<28}"
            for name, _ in STAGES:
                header += f" {name:>10}"
            lines.append(header)
            lines.append("  " + "─" * 110)

            for ds, total in DATASETS.items():
                for model in MODELS:
                    row = f"  {ds:<8}×{model:<17}"
                    for stage_name, pattern in STAGES:
                        p = BASE / pattern.format(ds=ds, model=model)
                        n = count_lines(p)
                        if n is None:
                            row += f" {'--':>10}"
                        elif n >= total:
                            row += f" {'✅':>10}"
                        else:
                            frac = n / total
                            row += f" {bar(frac, 6)} {100*frac:>3.0f}%"
                    lines.append(row)
                lines.append("")  # blank line between datasets

            lines.append("  " + "─" * 110)
            lines.append("")

            if first:
                for _ in range(len(lines)):
                    sys.stdout.write("\n")
                first = False
            else:
                sys.stdout.write(f"\033[{prev_lines}A")

            for ln in lines:
                sys.stdout.write("\033[2K" + ln + "\n")
            sys.stdout.flush()
            prev_lines = len(lines)

            time.sleep(15)
    except KeyboardInterrupt:
        sys.stdout.write("\n  stopped by user\n")


if __name__ == "__main__":
    main()
