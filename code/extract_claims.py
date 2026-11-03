#!/usr/bin/env python3
"""Stage B.2: Extract atomic factual claims per response for U-ETF's S^(2). (H200-optimized)

Phase 0 audit note (github.com/tiejin98/MSUQ, extract_claims.py):
    The official claim extractor prompts an auxiliary LLM (qwen:32b via
    Ollama) with a fixed FActScore-style instruction + two few-shot examples,
    and requires the model to answer as a JSON list of {"claim": ...}
    dictionaries. We reproduce that prompt verbatim below. The official code
    joins all extracted claims for one response into a single string
    (claim_1.claim_2. ...) which then gets treated exactly like a "response"
    by the same NLI machinery used for S^(1) -- see claim_nli.py.

H200 optimization note (this revision): the original made one BLOCKING
Ollama call per response, strictly sequentially (500 sequential HTTP
round-trips for a 25-query x 20-sample set; on the 12GB card this ran at
~60-90s per response because qwen:32b itself was CPU-offloaded). Two
changes:
  1. --workers N issues N claim-extraction requests concurrently via a
     thread pool. Results are reassembled BY INDEX, so output order is
     identical to the sequential version regardless of completion order.
  2. On a 4xH200 box, Ollama can hold qwen:32b entirely in VRAM, so the
     per-request latency collapses and concurrency is what determines
     throughput. Set OLLAMA_NUM_PARALLEL on the server to at least
     --workers, otherwise Ollama queues the extra requests and the
     parallelism is lost. Start conservative (--workers 8) and raise it
     while watching `ollama ps`.

Determinism note: claim extraction is an LLM call with no seed exposed
through this path, so it was never bit-reproducible, sequential or not.
Parallelism does not change that -- but because results are reassembled by
index, it does not add any NEW nondeterminism to the output ordering either.

This module is deliberately independent of correctness labels, gold answers,
and the U-ETF reliability/gating layer.

Input:  Stage-A JSONL (id, question, responses[].text)
Output: JSONL with one extra field per response: "claims_text" (the
        official-format joined-claims string) plus "claims" (parsed list,
        kept for inspection/debugging -- not required downstream).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

LOGGER = logging.getLogger("uetf.extract_claims")
DEFAULT_MODEL = "qwen:32b"

# Verbatim (mod-whitespace-only) from the official repo's extract_combined_claims.
_CLAIM_PROMPT_TEMPLATE = (
    "<<INST>><<SYS>>\n"
    "You are given a piece of text that includes knowledge claims. A claim is "
    "a statement that asserts something as true or false, which can be "
    "verified by humans. [Task] Your task is to accurately identify and "
    "extract every claim stated in the provided text. Then, resolve any "
    "coreference (pronouns or other referring expressions) in the claim for "
    "clarity. Each claim should be concise (less than 15 words) and "
    "self-contained. Your response MUST be a list of dictionaries. Each "
    "dictionary should contains the key \"claim\", which correspond to the "
    "extracted claim (with all coreferences resolved). You MUST only respond "
    "in the format as described below. DO NOT RESPOND WITH ANYTHING ELSE. "
    "ADDING ANY OTHER EXTRA RESPONSE FORMAT IS BANNED INCLUDING EXPLANATION. "
    "START YOUR RESPONSE WITH '['. [Response Format] "
    "[{{\"claim\": \"Ensure that the claim is fewer than 15 words and conveys "
    "a complete idea. Resolve any coreference (pronouns or other referring "
    "expressions) in the claim for clarity.\" }},... ] "
    "Here are two examples: "
    "[text]: Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The "
    "sixth-seed reaches Monte Carlo Masters final for the first time . "
    "Berdych will face either Rafael Nadal or Novak Djokovic in the final. "
    "[response]: [{{\"claim\": \"Tomas Berdych defeated Gael Monfis 6-1, "
    "6-4\"}}, {{\"claim\": \"Tomas Berdych defeated Gael Monfis 6-1, 6-4 on "
    "Saturday\"}}, {{\"claim\": \"Tomas Berdych reaches Monte Carlo Masters "
    "final\"}}, {{\"claim\": \"Tomas Berdych is the sixth-seed\"}}, "
    "{{\"claim\": \"Tomas Berdych reaches Monte Carlo Masters final for the "
    "first time\"}}, {{\"claim\": \"Berdych will face either Rafael Nadal or "
    "Novak Djokovic\"}}, {{\"claim\": \"Berdych will face either Rafael Nadal "
    "or Novak Djokovic in the final\"}}] "
    "[text]: Tinder only displays the last 34 photos - but users can easily "
    "see more. Firm also said it had improved its mutual friends feature. "
    "[response]: [{{\"claim\": \"Tinder only displays the last photos\"}}, "
    "{{\"claim\": \"Tinder only displays the last 34 photos\"}}, {{\"claim\": "
    "\"Tinder users can easily see more photos\"}}, {{\"claim\": \"Tinder "
    "said it had improved its feature\"}}, {{\"claim\": \"Tinder said it had "
    "improved its mutual friends feature\"}}] "
    "Now complete the following: "
    "[text]: Question:{question} Answer:{answer} "
    "[response]: [/INST]"
)

_UNWANTED_ANSWERS = {"", " ", ".", "{}", "[]", "()", "null", "none", "undefined"}


def is_unwanted_answer(text: str) -> bool:
    return text.strip().lower() in _UNWANTED_ANSWERS


def query_ollama(prompt: str, model: str) -> str:
    if ollama is None:
        raise ImportError("Please install the 'ollama' package: pip install ollama")
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"main_gpu": 0},
    )
    return response["message"]["content"]


def extract_claims_for_response(
    question: str,
    answer: str,
    *,
    model: str,
    max_retries: int = 5,
) -> list[str]:
    """Extract atomic claims for one response, retrying on malformed JSON."""
    if is_unwanted_answer(answer):
        return []

    prompt = _CLAIM_PROMPT_TEMPLATE.format(question=question, answer=answer)
    attempts = 0
    while attempts <= max_retries:
        raw = query_ollama(prompt, model)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Response JSON is not a list.")
            return [str(item["claim"]) for item in parsed if "claim" in item]
        except Exception as exc:  # noqa: BLE001 - LLM output can be malformed
            attempts += 1
            LOGGER.warning(
                "Claim extraction retry %d/%d (question=%r): %s",
                attempts, max_retries, question[:60], exc,
            )
            time.sleep(0.5)
    LOGGER.error("Giving up on claim extraction after %d retries; storing no claims.", max_retries)
    return []


def extract_claims_for_query(
    question: str,
    answers: list[str],
    *,
    model: str,
    max_retries: int,
    workers: int,
    query_id: str,
) -> list[list[str]]:
    """Extract claims for every response of one query.

    With workers > 1 the requests are issued concurrently, but results are
    collected by index so the returned list is in the same order as
    `answers` -- identical to what the sequential path would produce.
    """
    if workers <= 1:
        results = []
        for idx, answer in enumerate(answers):
            claims = extract_claims_for_response(question, answer, model=model, max_retries=max_retries)
            results.append(claims)
            LOGGER.info("query=%s response=%d/%d claims=%d", query_id, idx + 1, len(answers), len(claims))
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                lambda answer: extract_claims_for_response(
                    question, answer, model=model, max_retries=max_retries
                ),
                answers,
            )
        )
    LOGGER.info(
        "query=%s responses=%d claims_min/mean/max=%d/%.1f/%d (workers=%d)",
        query_id, len(answers),
        min(len(c) for c in results),
        sum(len(c) for c in results) / len(results),
        max(len(c) for c in results),
        workers,
    )
    return results


def claims_to_text(claims: list[str]) -> str:
    """Official format: claims joined with '.' -- matches extract_combined_claims."""
    return "".join(f"{c}." for c in claims)


def validate_responses(record: dict[str, Any], line_no: int) -> tuple[str, str, list[str]]:
    if "id" not in record or "question" not in record:
        raise ValueError(f"Input line {line_no} must contain 'id' and 'question'.")
    responses = record.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"Input line {line_no} must contain a non-empty 'responses' list.")
    texts = [str(r["text"]).strip() for r in responses]
    return str(record["id"]), str(record["question"]), texts


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    limit: int | None,
    max_retries: int,
    workers: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            if limit is not None and processed >= limit:
                break

            record = json.loads(line)
            query_id, question, texts = validate_responses(record, line_no)

            claims_per_response = extract_claims_for_query(
                question, texts, model=model, max_retries=max_retries, workers=workers, query_id=query_id
            )

            result = {
                "id": query_id,
                "question": question,
                "responses": record["responses"],
                "claims": {
                    "extractor_model": model,
                    "workers": workers,
                    "per_response": [
                        {
                            "sample_id": record["responses"][i].get("sample_id", i),
                            "claims": claims_per_response[i],
                            "claims_text": claims_to_text(claims_per_response[i]),
                        }
                        for i in range(len(texts))
                    ],
                },
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for claim extraction")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent Ollama requests (1 = old sequential behavior). Set the server's "
        "OLLAMA_NUM_PARALLEL to at least this value or the requests just queue.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    process_file(
        args.input, args.output,
        model=args.model, limit=args.limit, max_retries=args.max_retries, workers=args.workers,
    )


if __name__ == "__main__":
    main()
