#!/usr/bin/env python3
"""Stage A: Black-box response generation for U-ETF.

Supports both Hugging Face local models and Ollama models via --use-ollama.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# New import for Ollama
try:
    import ollama
except ImportError:
    ollama = None

LOGGER = logging.getLogger("uetf.generate")


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    num_samples: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 256
    seed: int = 20260903
    use_ollama: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, use_ollama: bool = False):
    """Load tokenizer/model for local black-box generation."""
    if use_ollama:
        if ollama is None:
            raise ImportError("Please install 'ollama' library: pip install ollama")
        # No local model loading; we will call ollama API directly
        return None, None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map: str | None = "auto" if torch.cuda.is_available() else None

        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()
        return tokenizer, model


def build_prompt(tokenizer, question: str) -> str:
    """Use model's chat template when available; otherwise plain text."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def generate_one(
    tokenizer,
    model,
    question: str,
    *,
    generation: GenerationConfig,
    sample_seed: int,
) -> str:
    """Generate one stochastic response (supports both transformers and Ollama)."""
    set_seed(sample_seed)  # for reproducibility (Ollama may not fully honour seed)

    if generation.use_ollama:
        # Use Ollama's chat API (or generate) with options
        # Note: seed is not officially supported by Ollama, but we pass it anyway.
        options = {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "top_k": generation.top_k,
            "num_predict": generation.max_new_tokens,
            "seed": sample_seed,  # some models may respect it
        }
        # Use chat template if possible; otherwise simple prompt
        # Ollama's default chat template expects messages list
        messages = [{"role": "user", "content": question}]
        response = ollama.chat(
            model=generation.model,
            messages=messages,
            options=options,
        )
        return response["message"]["content"].strip()
    else:
        # Original transformers-based generation
        prompt = build_prompt(tokenizer, question)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_k=generation.top_k,
                max_new_tokens=generation.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output[0, prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text


def validate_query(record: dict[str, Any], line_no: int) -> tuple[str, str]:
    if "id" not in record or "question" not in record:
        raise ValueError(f"Input line {line_no} must contain 'id' and 'question'.")
    query_id = str(record["id"]).strip()
    question = str(record["question"]).strip()
    if not query_id or not question:
        raise ValueError(f"Input line {line_no} has an empty id or question.")
    return query_id, question


def generate_dataset(
    input_path: Path,
    output_path: Path,
    generation: GenerationConfig,
    limit: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(generation.model, use_ollama=generation.use_ollama)
    LOGGER.info("Using model: %s (Ollama: %s)", generation.model, generation.use_ollama)

    processed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id, question = validate_query(record, line_no)

            if limit is not None and processed >= limit:
                break

            responses: list[dict[str, Any]] = []
            for sample_id in range(generation.num_samples):
                sample_seed = generation.seed + processed * 1000 + sample_id
                text = generate_one(
                    tokenizer,
                    model,
                    question,
                    generation=generation,
                    sample_seed=sample_seed,
                )
                responses.append(
                    {
                        "sample_id": sample_id,
                        "text": text,
                        "seed": sample_seed,
                    }
                )

                LOGGER.info(
                    "query=%s sample=%d/%d",
                    query_id,
                    sample_id + 1,
                    generation.num_samples,
                )

            result = {
                "id": query_id,
                "question": question,
                "responses": responses,
                "generation": asdict(generation),
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1

    LOGGER.info("Completed %d queries. Output: %s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name (HF or Ollama)")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N queries")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama instead of HF transformers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    generation = GenerationConfig(
        model=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        use_ollama=args.use_ollama,
    )

    generate_dataset(args.input, args.output, generation, limit=args.limit)


if __name__ == "__main__":
    main()