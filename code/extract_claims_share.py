#!/usr/bin/env python3
"""Stage B.2: Extract atomic factual claims per response for U-ETF's S^(2). (H200-optimized)

Supports --ollama-host for multi-GPU parallelism (run multiple Ollama
instances on different GPUs/ports, one extract_claims process per host).
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
except ImportError:
    ollama = None

LOGGER = logging.getLogger("uetf.extract_claims")
DEFAULT_MODEL = "qwen:32b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

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

# Global ollama client — set in main()
_OLLAMA_CLIENT = None


def is_unwanted_answer(text: str) -> bool:
    return text.strip().lower() in _UNWANTED_ANSWERS


def query_ollama(prompt: str, model: str) -> str:
    if _OLLAMA_CLIENT is None:
        raise RuntimeError("Ollama client not initialized.")
    response = _OLLAMA_CLIENT.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def extract_claims_for_response(
    question: str,
    answer: str,
    *,
    model: str,
    max_retries: int = 5,
) -> list[str]:
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
        except Exception as exc:
            attempts += 1
            LOGGER.warning("Claim extraction retry %d/%d: %s", attempts, max_retries, exc)
            time.sleep(0.5)
    LOGGER.error("Giving up on claim extraction after %d retries.", max_retries)
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
    if workers <= 1:
        results = []
        for idx, answer in enumerate(answers):
            claims = extract_claims_for_response(question, answer, model=model, max_retries=max_retries)
            results.append(claims)
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
    return results


def claims_to_text(claims: list[str]) -> str:
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
    resume: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    file_mode = "a" if resume else "w"

    processed_ids = set()
    if resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(str(json.loads(line)["id"]))
                    except Exception:
                        pass
        LOGGER.info("Resume: %d IDs already processed.", len(processed_ids))

    with input_path.open("r", encoding="utf-8") as src, output_path.open(file_mode, encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = str(record["id"])
            if resume and query_id in processed_ids:
                continue
            if limit is not None and processed >= limit:
                break

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
            if processed % 20 == 0:
                LOGGER.info("progress: processed=%d", processed)

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST,
                        help="Ollama endpoint (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Skip IDs already present in the output file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    global _OLLAMA_CLIENT
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if ollama is None:
        raise ImportError("Please install the 'ollama' package.")

    _OLLAMA_CLIENT = ollama.Client(host=args.ollama_host)
    LOGGER.info("Using Ollama host: %s", args.ollama_host)

    process_file(
        args.input, args.output,
        model=args.model, limit=args.limit,
        max_retries=args.max_retries, workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
