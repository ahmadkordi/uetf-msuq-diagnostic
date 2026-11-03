#!/usr/bin/env python3
"""Monitor running score_correctness.py processes: their progress, rate, and ETA."""
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def running_procs():
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,etime,args"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
    except Exception:
        return []
    procs = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if "score_correctness.py" not in line or "grep" in line:
            continue
        pid_m = re.match(r"(\d+)\s+(\S+)\s+", line)
        if not pid_m:
            continue
        pid, etime = pid_m.groups()

        out_m = re.search(r"--output\s+(\S+)", line)
        in_m = re.search(r"--input\s+(\S+)", line)
        port_m = re.search(r"114\d\d", line)
        if not out_m:
            continue
        procs.append({
            "pid": pid,
            "etime": etime,
            "output": Path(out_m.group(1)),
            "input": Path(in_m.group(1)) if in_m else None,
            "port": port_m.group(0) if port_m else "?",
        })
    return procs


def count_lines(p):
    if not p.exists():
        return 0
    n = 0
    try:
        with p.open("rb") as f:
            for _ in f:
                n += 1
    except Exception:
        return 0
    return n


def input_size(p):
    if not p or not p.exists():
        return 0
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def fmt_dur(s):
    if s is None or s < 0:
        return "--:--:--"
    s = int(s)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h > 99:
        return f"{h}h{m:02d}m"
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main():
    hist = {}
    first = True
    prev_lines = 0
    while True:
        procs = running_procs()
        now = time.time()

        lines = []
        lines.append(f"  score_correctness monitor  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("  " + "=" * 118)
        lines.append(f"  {'PID':>7}  {'Elapsed':>9}  {'Port':>5}  {'Output':<48}  {'Lines':>12}  {'Prog':>6}  {'Rate':>9}  {'ETA':>9}")
        lines.append("  " + "-" * 118)

        if not procs:
            lines.append("  (no score_correctness processes running)")
        for p in procs:
            key = str(p["output"])
            n = count_lines(p["output"])
            total = input_size(p["input"])

            hist.setdefault(key, deque(maxlen=20))
            hist[key].append((now, n))
            h = hist[key]
            rate = 0.0
            if len(h) >= 2:
                t0, n0 = h[0]
                t1, n1 = h[-1]
                dt = t1 - t0
                if dt > 0:
                    rate = (n1 - n0) / dt  # lines/sec

            eta_s = (total - n) / rate if rate > 0 and total > 0 else None
            frac = n / total if total > 0 else 0.0
            pct = 100.0 * frac

            name = p["output"].name
            if len(name) > 48:
                name = name[:45] + "..."

            lines.append(
                f"  {p['pid']:>7}  {p['etime']:>9}  {p['port']:>5}  {name:<48}  {n:>5}/{total:<5}  {pct:>5.1f}%  {rate*60:>6.1f}/m  {fmt_dur(eta_s):>9}"
            )

        lines.append("  " + "-" * 118)
        lines.append(f"  Active: {len(procs)} process(es)")
        lines.append("")

        if first:
            for _ in lines:
                sys.stdout.write("\n")
            first = False
        else:
            sys.stdout.write(f"\033[{prev_lines}A")

        for ln in lines:
            sys.stdout.write("\033[2K" + ln + "\n")
        sys.stdout.flush()
        prev_lines = len(lines)

        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\n  stopped\n")
