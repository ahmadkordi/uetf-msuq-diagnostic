#!/usr/bin/env python3
import argparse
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path


def count_lines(path):
    n = 0
    if path.exists():
        with path.open("rb") as f:
            for _ in f:
                n += 1
    return n


def fmt_dur(seconds):
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--:--:--"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 99:
        return f"{h}h{m:02d}m"
    return f"{h:02d}:{m:02d}:{s:02d}"


def bar(frac, width=30):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def parse_job(spec):
    name, rest = spec.split("=", 1)
    path, total = rest.rsplit(":", 1)
    return name.strip(), Path(path.strip()), int(total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", action="append", required=True)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()

    jobs = [parse_job(s) for s in args.job]
    hist = {name: deque(maxlen=args.window) for name, _, _ in jobs}
    start_time = time.time()
    lines_per_job = 4
    header_lines = 2
    prev_lines = 0
    first_draw = True

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            block = []
            block.append(f"  U-ETF Monitor  |  elapsed {fmt_dur(elapsed)}  |  {datetime.now().strftime('%H:%M:%S')}")
            block.append("  " + "-" * 78)

            all_done = True
            for name, path, total in jobs:
                n = count_lines(path)
                hist[name].append((now, n))
                h = hist[name]

                rate = 0.0
                if len(h) >= 2:
                    t0, n0 = h[0]
                    t1, n1 = h[-1]
                    dt = t1 - t0
                    if dt > 0:
                        rate = (n1 - n0) / dt

                frac = n / total if total else 0.0
                eta_s = (total - n) / rate if rate > 0 else None
                eta_dt = datetime.now() + timedelta(seconds=eta_s) if eta_s is not None else None
                status = "DONE" if n >= total else "running"
                if n < total:
                    all_done = False

                block.append(f"  [{name}] {status}")
                block.append(f"    Progress  [{bar(frac)}]  {n:>6}/{total}  ({100*frac:5.1f}%)")
                block.append(
                    f"    Rate      {rate*60:6.2f} q/min    "
                    f"ETA       {fmt_dur(eta_s):>8s}    "
                    f"Finish at {eta_dt.strftime('%H:%M:%S') if eta_dt else '--'}"
                )
                block.append("")

            if first_draw:
                for _ in range(len(block)):
                    sys.stdout.write("\n")
                first_draw = False
            else:
                sys.stdout.write(f"\033[{prev_lines}A")

            for ln in block:
                sys.stdout.write("\033[2K" + ln + "\n")
            sys.stdout.flush()
            prev_lines = len(block)

            if all_done:
                sys.stdout.write("\n  All jobs DONE\n")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        sys.stdout.write("\n  stopped by user\n")


if __name__ == "__main__":
    main()
