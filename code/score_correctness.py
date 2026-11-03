#!/usr/bin/env python3
"""Stage Eval: Correctness scoring for U-ETF (Experiment 0B/0C/0D prerequisite).

Phase 0 audit note (github.com/tiejin98/MSUQ, correctness_qwen.py --
verbatim source re-fetched and verified in this revision via
https://raw.githubusercontent.com/tiejin98/MSUQ/main/correctness_qwen.py):

    The official judge is Qwen2.5-32B-Instruct, served with vllm,
    `gpu_memory_utilization=0.98`. We substitute Ollama here (same
    documented deviation already used for extract_claims.py).

    1. Prompt content. The official prompt has NO "additional acceptable
       answers" field at all -- it only ever compares one "original answer"
       against one "provided answer". Verbatim official template
       (`build_prompt`):

           Given the following question, the original answer, and a
           provided answer, output an integer from 0 to 100 (no other
           text) indicating how correct the provided answer is.

           Question:
           {prompt_text}

           Original Answer:
           {original_ans}

           Provided Answer:
           {provided_ans}

           correctness:

    2. Full wire format: flat string, no chat template:
           "System: You are a helpful assistant. And you should only "
           "output numbers to present correctness.\\n"
           f"User: {user_prompt}\\n"

    3. Sampling: temperature=0.0, max_tokens=10, top_p=1.0, n=1.

    4. Parsing: `re.search(r"\\d+")`, clamped to [0, 100], 0 if no match.

    5. Threshold: `> 90` = correct (verified in official aggregation).

Input:  any of the Stage-A outputs -- anything with responses[].text
        and either gold_answer (CoQA/NQ-Open/HotpotQA) or best_answer
        (TruthfulQA).
Output: JSONL with, per query: each response's raw judge score (0-100,
        float) and binary C(y_i) = 1[score > 90], plus the aggregated
        U_N_star(q) = 1 - mean(C) needed for Experiment 0B's z_k^abs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

LOGGER = logging.getLogger("uetf.score_correctness")
DEFAULT_MODEL = "qwen2.5:32b"
CORRECT_THRESHOLD = 90  # verbatim from official correctness_qwen.py


def build_prompt_text(record: dict[str, Any]) -> str:
    """Approximate the official code's `tokenizer.decode(item['prompt'])`."""
    story = record.get("story_so_far") or record.get("story") or record.get("passage")
    question = str(record["question"]).strip()
    if story:
        return f"{str(story).strip()} Q: {question}"
    return question


def build_prompt(prompt_text: str, original_ans: str, provided_ans: str) -> str:
    """Verbatim port of the official `build_prompt` (correctness_qwen.py)."""
    user_prompt = f"""
Given the following question, the original answer, and a provided answer, output an integer from 0 to 100 (no other text) indicating how correct the provided answer is.

Question:
{prompt_text}

Original Answer:
{original_ans}

Provided Answer:
{provided_ans}

correctness:
""".strip()
    return user_prompt


def build_wire_prompt(user_prompt: str) -> str:
    """Verbatim port of the official flat System/User wrapping (no chat template)."""
    return (
        "System: You are a helpful assistant. "
        "And you should only output numbers to present correctness.\n"
        f"User: {user_prompt}\n"
    )


def parse_score(generated_text: str) -> float:
    """Verbatim port of the official `parse_correctness_score`."""
    match = re.search(r"\d+", generated_text)
    if match is None:
        return 0.0
    value = int(match.group(0))
    return float(max(0, min(100, value)))


def query_judge(prompt: str, model: str) -> str:
    if ollama is None:
        raise ImportError("Please install the 'ollama' package: pip install ollama")
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.0, "top_p": 1.0, "num_predict": 10, "main_gpu": 0},
    )
    return response["response"]


def score_one_response(prompt_text: str, original_ans: str, provided_ans: str, *, model: str) -> float:
    user_prompt = build_prompt(prompt_text, original_ans, provided_ans)
    wire_prompt = build_wire_prompt(user_prompt)
    generated_text = query_judge(wire_prompt, model).strip()
    return parse_score(generated_text)


def score_query(
    prompt_text: str, original_ans: str, candidates: list[str], *, model: str, workers: int, query_id: str,
) -> list[float]:
    fn = lambda c: score_one_response(prompt_text, original_ans, c, model=model)
    if workers <= 1:
        scores = []
        for idx, c in enumerate(candidates):
            s = fn(c)
            scores.append(s)
            LOGGER.info("query=%s response=%d/%d score=%s", query_id, idx + 1, len(candidates), s)
        return scores
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(fn, candidates))
    LOGGER.info("query=%s responses=%d scored (workers=%d)", query_id, len(candidates), workers)
    return scores


def validate_record(record: dict[str, Any], line_no: int) -> tuple[str, str, str, list[str]]:
    if "id" not in record or "question" not in record:
        raise ValueError(f"Input line {line_no} must contain 'id' and 'question'.")
    # TruthfulQA Stage-A output stores `best_answer`, not `gold_answer`.
    gold = record.get("gold_answer") or record.get("best_answer")
    if not gold:
        raise ValueError(
            f"Input line {line_no} has neither 'gold_answer' nor 'best_answer' -- this script "
            "needs the Stage-A output directly."
        )
    responses = record.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"Input line {line_no} must contain a non-empty 'responses' list.")
    candidates = [str(r["text"]).strip() for r in responses]
    return str(record["id"]), str(record["question"]), str(gold), candidates


def process_file(
    input_path: Path, output_path: Path, *,
    model: str, limit: int | None, workers: int, resume: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    processed_ids: set[str] = set()
    file_mode = "w"
    if resume and output_path.exists():
        file_mode = "a"
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        LOGGER.info("Resume: %d IDs already processed, appending to %s", len(processed_ids), output_path)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(file_mode, encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid_raw = str(record.get("id", "")).strip()
            if resume and qid_raw in processed_ids:
                continue
            if limit is not None and processed >= limit:
                break

            query_id, question, gold, candidates = validate_record(record, line_no)
            prompt_text = build_prompt_text(record)

            scores = score_query(
                prompt_text, gold, candidates,
                model=model, workers=workers, query_id=query_id,
            )
            correct = [1 if s > CORRECT_THRESHOLD else 0 for s in scores]
            u_n_star = 1.0 - (sum(correct) / len(correct))

            result = {
                "id": query_id,
                "question": question,
                "judge_model": model,
                "threshold": CORRECT_THRESHOLD,
                "per_response": [
                    {"sample_id": record["responses"][i].get("sample_id", i), "score": scores[i], "correct": correct[i]}
                    for i in range(len(candidates))
                ],
                "U_N_star": u_n_star,
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            LOGGER.info("query=%s U_N_star=%.4f", query_id, u_n_star)

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Stage-A JSONL (must have gold_answer or best_answer)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama judge model")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent judge calls; set server OLLAMA_NUM_PARALLEL >= this")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in output and append.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    process_file(
        args.input, args.output,
        model=args.model, limit=args.limit,
        workers=args.workers, resume=args.resume,
    )


if __name__ == "__main__":
    main()
