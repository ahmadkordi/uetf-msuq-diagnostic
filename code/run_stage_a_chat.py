#!/usr/bin/env python3
"""Stage A for Llama-2-13B-CHAT: sharded generation scheduled over the GPUs from Python (no shell script).

It runs the SAME generation scripts that produced the base runs (raw completion, T=1.0, top_p=1.0, top_k=0, 256 new
tokens, N=20, bf16, batched sampling, seed 20260903); only --model differs. No sampling flag is passed, so the
scripts' own defaults apply. Each (dataset, shard) is one subprocess pinned to one GPU with --gpu-id.

  python run_stage_a_chat.py smoke  --coqa-file /path/coqa-dev-v1.0.json     # 3 queries per dataset, inspect first
  python run_stage_a_chat.py run    --coqa-file /path/coqa-dev-v1.0.json     # all shards, then merge
  python run_stage_a_chat.py status
  python run_stage_a_chat.py merge                                           # re-run only the merge/validation

Shard files:  <work-dir>/<dataset>/<dataset>.shard{k}of{n}.jsonl   (+ .done marker when validated)
Merged file:  <out-dir>/stage_a_<dataset>_Llama2chat_13b.jsonl       (written atomically, only if all shards are valid)
Merge order = dataset order (record g = k + j*n is the j-th line of shard k); the same file name is what
run_chat_pipeline.sh expects, so its stageA then skips generation and goes straight to the judge.

Known properties of the underlying scripts (not changed here):
  * per-query seed = seed + (index WITHIN the shard)*1000, so seeds are not unique across shards and depend on
    the number of shards; outputs are statistically equivalent to an unsharded run but not bit-identical;
  * only the HotpotQA script has --resume (used automatically, after trimming a half-written last line);
    for the other datasets an interrupted shard is restarted from scratch (its partial file is renamed, not deleted);
  * the scripts keep a "Stopping config" line in the log; smoke prints it so it can be compared with the base logs.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ORDER = ["hotpotqa", "coqa", "nqopen", "truthfulqa"]          # scheduling priority (heavy first)
TOTAL = {"coqa": 7983, "hotpotqa": 7405, "nqopen": 3610, "truthfulqa": 817}
SCRIPT = {
    "coqa": "generate_responses_coqa_shard.py",
    "hotpotqa": "generate_responses_hotpotqa_v3_share_task.py",
    "nqopen": "generate_responses_nqopen_shard.py",
    "truthfulqa": "generate_responses_truthfulqa_shard.py",
}
BASE_CELL = {d: f"{d}_Llama2_13b" for d in TOTAL}
CHAT_CELL = {d: f"{d}_Llama2chat_13b" for d in TOTAL}
DEFAULT_SHARDS = {"coqa": 8, "hotpotqa": 8, "nqopen": 4, "truthfulqa": 4}
CHATTER = ("sure", "certainly", "of course", "thank", "i'm ", "i am ", "as an ai", "you are correct", "great", "yes, ")

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- helpers
def expected_count(ds: str, k: int, n: int) -> int:
    return len(range(k, TOTAL[ds], n))


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("rb") as fh:
        return sum(1 for ln in fh if ln.strip())


def repair_tail(p: Path) -> int:
    """Drop a trailing half-written / unparsable line so that --resume appends cleanly. Returns number dropped."""
    if not p.exists():
        return 0
    parts = p.read_bytes().split(b"\n")
    frag, keep = parts[-1], parts[:-1]          # frag == b"" when the file ends with a newline
    dropped = 1 if frag.strip() else 0
    while keep:
        try:
            json.loads(keep[-1])
            break
        except Exception:
            keep.pop(); dropped += 1
    if dropped:
        p.write_bytes((b"\n".join(keep) + b"\n") if keep else b"")
    return dropped


class Job:
    def __init__(self, ds: str, k: int, n: int, work: Path):
        self.ds, self.k, self.n = ds, k, n
        self.dir = work / ds
        self.path = self.dir / f"{ds}.shard{k}of{n}.jsonl"
        self.marker = Path(str(self.path) + ".done")
        self.log = self.dir / "logs" / f"{ds}.shard{k}of{n}.log"
        self.expected = expected_count(ds, k, n)

    def name(self) -> str:
        return f"{self.ds}[{self.k}/{self.n}]"

    def is_done(self) -> bool:
        return self.marker.exists() and count_lines(self.path) == self.expected


def validate_shard(job: Job, num_samples: int):
    ids, bad = set(), 0
    n = 0
    with job.path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                return False, "unparsable line", None
            n += 1
            ids.add(str(r.get("id")))
            if len(r.get("responses", [])) != num_samples:
                bad += 1
    if n != job.expected:
        return False, f"{n} records, expected {job.expected}", None
    if len(ids) != n:
        return False, f"duplicate ids ({n - len(ids)})", None
    if bad:
        return False, f"{bad} records without {num_samples} responses", None
    return True, "ok", ids


def build_cmd(job: Job, gpu: int, a, out: Path, limit: int | None = None) -> list[str]:
    cmd = [sys.executable, str(Path(a.src) / SCRIPT[job.ds]), "--model", a.model, "--output", str(out),
           "--num-samples", str(a.num_samples), "--gpu-id", str(gpu), "--dtype", a.dtype,
           "--shard-id", str(job.k), "--num-shards", str(job.n)]
    if job.ds == "coqa":
        cmd += ["--coqa-file", a.coqa_file]
    elif job.ds == "hotpotqa":
        cmd += ["--hotpot-dir", a.hotpot_dir, "--resume"]
    elif job.ds == "nqopen":
        cmd += ["--nqopen-dir", a.nqopen_dir]
    elif job.ds == "truthfulqa":
        cmd += ["--truthfulqa-dir", a.truthfulqa_dir]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


ENV = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")


def preflight(a, datasets) -> bool:
    ok = True
    if not Path(a.model).is_dir():
        say(f"ERROR model dir not found: {a.model}"); ok = False
    for d in datasets:
        if not (Path(a.src) / SCRIPT[d]).exists():
            say(f"ERROR script missing: {Path(a.src) / SCRIPT[d]}"); ok = False
    if "coqa" in datasets and (not a.coqa_file or not Path(a.coqa_file).exists()):
        say("ERROR --coqa-file must point to coqa-dev-v1.0.json"); ok = False
    return ok


# ----------------------------------------------------------------------------- run
def run_job(job: Job, gpu: int, a, procs: set, stop: threading.Event) -> bool:
    job.dir.mkdir(parents=True, exist_ok=True); job.log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(a.max_retries + 1):
        if stop.is_set():
            return False
        if job.path.exists() and not job.marker.exists():
            if job.ds == "hotpotqa":
                d = repair_tail(job.path)
                say(f"{job.name()} resuming from {count_lines(job.path)} lines (trimmed {d} broken line(s))")
            else:
                new = job.path.with_name(job.path.name + f".partial-{int(time.time())}")
                job.path.rename(new)
                say(f"{job.name()} partial file kept as {new.name}; restarting shard from scratch")
        cmd = build_cmd(job, gpu, a, job.path)
        with job.log.open("ab") as lf:
            lf.write(f"\n=== attempt {attempt} gpu {gpu} {time.strftime('%F %T')}\n{' '.join(cmd)}\n".encode()); lf.flush()
            say(f"START {job.name()} on GPU {gpu} (attempt {attempt + 1})")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
            procs.add(p); rc = p.wait(); procs.discard(p)
        if stop.is_set():
            return False
        if rc == 0:
            ok, msg, _ = validate_shard(job, a.num_samples)
            if ok:
                job.marker.write_text(str(job.expected))
                say(f"DONE  {job.name()} ({job.expected} records)")
                return True
            say(f"FAIL  {job.name()} validation: {msg}")
        else:
            say(f"FAIL  {job.name()} exit code {rc} (see {job.log})")
    return False


def progress_loop(jobs, stop: threading.Event, every: int) -> None:
    t0 = time.time()
    while not stop.wait(every):
        parts = []
        for ds in ORDER:
            js = [j for j in jobs if j.ds == ds]
            if js:
                parts.append(f"{ds} {sum(count_lines(j.path) for j in js)}/{TOTAL[ds]}")
        say(f"progress ({(time.time() - t0) / 60:.0f} min): " + " | ".join(parts))


def make_jobs(a, datasets) -> list[Job]:
    shards = dict(DEFAULT_SHARDS)
    for item in (a.shards or "").split(","):
        if item.strip():
            k, v = item.split("="); shards[k.strip()] = int(v)
    jobs = []
    for d in ORDER:
        if d in datasets:
            jobs += [Job(d, k, shards[d], Path(a.work_dir)) for k in range(shards[d])]
    return jobs


def cmd_run(a) -> int:
    datasets = [d for d in ORDER if d in a.datasets]
    if not preflight(a, datasets):
        return 2
    jobs = make_jobs(a, datasets)
    pending = [j for j in jobs if not j.is_done()]
    say(f"{len(jobs)} shard jobs, {len(jobs) - len(pending)} already done, {len(pending)} to run on GPUs {a.gpus} x {a.jobs_per_gpu}")
    q: queue.Queue = queue.Queue()
    for j in pending:
        q.put(j)
    stop, procs, failed = threading.Event(), set(), []

    def slot(gpu: int):
        while not stop.is_set():
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            if not run_job(job, gpu, a, procs, stop):
                failed.append(job.name())

    threads = [threading.Thread(target=slot, args=(g,), daemon=True) for g in a.gpus for _ in range(a.jobs_per_gpu)]
    pt = threading.Thread(target=progress_loop, args=(jobs, stop, a.progress_every), daemon=True)
    try:
        for t in threads:
            t.start()
        pt.start()
        for t in threads:
            while t.is_alive():
                t.join(1.0)
    except KeyboardInterrupt:
        say("Ctrl-C: stopping children (finished shards are kept; rerun the same command to continue)")
        stop.set()
        for p in list(procs):
            p.terminate()
        return 130
    finally:
        stop.set()
    if failed:
        say("FAILED shards: " + ", ".join(failed) + "  -> rerun the same command to retry them")
        return 1
    say("all shards done")
    return 0 if a.no_merge else cmd_merge(a)


# ----------------------------------------------------------------------------- merge
def read_ids(p: Path) -> set:
    ids = set()
    with p.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                ids.add(str(json.loads(ln)["id"]))
    return ids


def resp_stats(p: Path, limit: int | None = None):
    lens, empty, chat, tot = [], 0, 0, 0
    with p.open("r", encoding="utf-8") as fh:
        for i, ln in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if not ln.strip():
                continue
            for r in json.loads(ln)["responses"]:
                t = r["text"].strip(); tot += 1
                lens.append(len(t)); empty += (len(t) == 0)
                chat += t.lower().startswith(CHATTER)
    if not tot:
        return None
    lens.sort()
    return dict(n=tot, empty=empty / tot, mean=statistics.mean(lens), med=lens[len(lens) // 2], p95=lens[int(0.95 * (len(lens) - 1))], chatter=chat / tot)


def cmd_merge(a) -> int:
    rc = 0
    for ds in [d for d in ORDER if d in a.datasets]:
        jobs = make_jobs(a, [ds])
        if not all(j.is_done() for j in jobs):
            say(f"merge {ds}: not all shards done ({sum(j.is_done() for j in jobs)}/{len(jobs)}) - skipped"); rc = 1; continue
        final = Path(a.out_dir) / f"stage_a_{CHAT_CELL[ds]}.jsonl"
        if final.exists() and final.stat().st_size > 0 and not a.force:
            say(f"merge {ds}: {final} already exists (use --force to overwrite) - skipped"); rc = 1; continue
        rows = []
        for j in jobs:
            with j.path.open("r", encoding="utf-8") as fh:
                for idx, ln in enumerate([x for x in fh if x.strip()]):
                    rows.append((j.k + idx * j.n, ln.rstrip("\n")))
        rows.sort(key=lambda x: x[0])
        gs = [g for g, _ in rows]
        ids = [str(json.loads(ln)["id"]) for _, ln in rows]
        if gs != list(range(TOTAL[ds])) or len(set(ids)) != TOTAL[ds]:
            say(f"merge {ds}: ERROR index/id check failed (n={len(rows)}, unique ids={len(set(ids))}, expected {TOTAL[ds]})"); rc = 1; continue
        base = Path(a.out_dir) / f"stage_a_{BASE_CELL[ds]}.jsonl"
        if base.exists():
            bids = read_ids(base)
            if bids != set(ids):
                say(f"merge {ds}: ERROR id set differs from base ({len(bids - set(ids))} only in base, {len(set(ids) - bids)} only in chat)")
                if not a.ignore_base_ids:
                    rc = 1; continue
            else:
                say(f"merge {ds}: id set identical to base ({len(bids)} ids)")
        else:
            say(f"merge {ds}: base file {base.name} not found - id comparison skipped")
        tmp = final.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for _, ln in rows:
                fh.write(ln + "\n")
        os.replace(tmp, final)
        st = resp_stats(final)
        bs = resp_stats(base) if base.exists() else None
        say(f"merge {ds}: wrote {final} ({len(rows)} records)")
        say(f"   chat: empty {st['empty']:.3%}  chars mean {st['mean']:.1f} median {st['med']} p95 {st['p95']}  chatter-prefix(heuristic) {st['chatter']:.2%}")
        if bs:
            say(f"   base: empty {bs['empty']:.3%}  chars mean {bs['mean']:.1f} median {bs['med']} p95 {bs['p95']}  chatter-prefix(heuristic) {bs['chatter']:.2%}")
    return rc


# ----------------------------------------------------------------------------- smoke / status
def cmd_smoke(a) -> int:
    datasets = [d for d in ORDER if d in a.datasets]
    if not preflight(a, datasets):
        return 2
    sdir = Path(a.work_dir) / "smoke"; sdir.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    def one(ds: str, gpu: int):
        n = make_jobs(a, [ds])[0].n
        job = Job(ds, 0, n, Path(a.work_dir))
        out = sdir / f"{ds}.jsonl"; log = sdir / f"{ds}.log"
        if out.exists():
            out.unlink()
        cmd = build_cmd(job, gpu, a, out, limit=a.smoke_n)
        if ds == "hotpotqa":
            cmd.remove("--resume")
        with log.open("wb") as lf:
            say(f"smoke {ds} on GPU {gpu}")
            res[ds] = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV).returncode

    ts = [threading.Thread(target=one, args=(d, a.gpus[i % len(a.gpus)])) for i, d in enumerate(datasets)]
    [t.start() for t in ts]; [t.join() for t in ts]
    for ds in datasets:
        print(f"\n===== {ds} (exit {res.get(ds)}) =====")
        log = sdir / f"{ds}.log"; out = sdir / f"{ds}.jsonl"
        if log.exists():
            txt = log.read_text(errors="replace").splitlines()
            for ln in txt:
                if "Stopping config" in ln or "not LlamaTokenizer" in ln or "Shard " in ln:
                    print("  LOG:", ln[-300:])
            if res.get(ds) != 0:
                print("  --- last log lines ---"); print("\n".join("  " + x for x in txt[-12:]))
        if out.exists() and out.stat().st_size:
            recs = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
            g = recs[0].get("generation", {})
            print("  generation field:", {k: g.get(k) for k in ("model", "dtype", "temperature", "top_p", "top_k", "max_new_tokens", "seed", "num_samples", "batched_sampling")})
            for r in recs[:2]:
                print("  Q:", r["question"][:120].replace("\n", " "))
                for x in r["responses"][:3]:
                    print("     ->", repr(x["text"][:200]))
            st = resp_stats(out)
            print(f"  stats over {st['n']} responses: empty {st['empty']:.1%}, mean chars {st['mean']:.1f}, chatter-prefix(heuristic) {st['chatter']:.1%}")
    return 0


def cmd_status(a) -> int:
    for ds in [d for d in ORDER if d in a.datasets]:
        js = make_jobs(a, [ds])
        print(f"{ds}: {sum(count_lines(j.path) for j in js)}/{TOTAL[ds]} lines, shards done {sum(j.is_done() for j in js)}/{len(js)}")
        for j in js:
            print(f"   {j.name():18s} {count_lines(j.path):5d}/{j.expected:<5d} {'DONE' if j.is_done() else ''}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["smoke", "run", "merge", "status"])
    ap.add_argument("--src", default="/mnt/src")
    ap.add_argument("--model", default="/mnt/data/models/Llama-2-13b-chat-hf")
    ap.add_argument("--work-dir", default="/mnt/data/tmp/chat_stage_a")
    ap.add_argument("--out-dir", default="/mnt/output")
    ap.add_argument("--datasets", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 3])
    ap.add_argument("--jobs-per-gpu", type=int, default=1, help="processes per GPU (default 1; 2 may raise throughput, untested)")
    ap.add_argument("--shards", default="", help="override, e.g. coqa=8,hotpotqa=8,nqopen=4,truthfulqa=4")
    ap.add_argument("--coqa-file", default=os.environ.get("COQA_FILE", ""))
    ap.add_argument("--hotpot-dir", default="/mnt/data/datasets/hotpotqa_local")
    ap.add_argument("--nqopen-dir", default="/mnt/data/datasets/nq_open_local")
    ap.add_argument("--truthfulqa-dir", default="/mnt/data/datasets/truthfulqa_local")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--progress-every", type=int, default=120)
    ap.add_argument("--smoke-n", type=int, default=3)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--force", action="store_true", help="merge: overwrite an existing final file")
    ap.add_argument("--ignore-base-ids", action="store_true")
    a = ap.parse_args()
    return {"smoke": cmd_smoke, "run": cmd_run, "merge": cmd_merge, "status": cmd_status}[a.action](a)


if __name__ == "__main__":
    sys.exit(main())
