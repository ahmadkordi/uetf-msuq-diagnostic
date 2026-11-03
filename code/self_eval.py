#!/usr/bin/env python3
"""Stage B.4: Build the self-evaluation relation matrix S^(3) for U-ETF. (H200-optimized)

Methodology v9 (this is a U-ETF DESIGN CHOICE, not an official MS-UQ
definition -- the official baseline only ever uses a single scalar
U_p = 1 - mean(p_true) per query, never a matrix):
    S^(3)_ii = P_true(y_i | q)
    S^(3)_ij = sqrt(P_true(y_i | q) * P_true(y_j | q))   for i != j

Phase 0 audit note (github.com/tiejin98/MSUQ, dataeval/load_worker.py,
_get_self_eval_sample):
    P_true is estimated by prompting the SAME model that generated the
    responses (not a separate judge) with a fixed multiple-choice template,
    then reading the raw next-token logits at the two candidate tokens
    "(A" / "(B" (True / False) and taking a 2-way softmax. This REQUIRES
    logit access to the generating model.

    The official prompt also injects up to 10 "brainstormed ideas" (other
    sampled answers) drawn from a Python set, whose iteration order is not
    guaranteed deterministic. We sort them instead -- a documented,
    deliberate improvement.

H200 optimization note (this revision): the original version made one
model() forward pass per response (e.g. 500 sequential calls for 25
queries x 20 responses). This version batches all N responses for a query
into ONE padded forward pass (left-padding, so the last real token is
always at position -1 for every row regardless of prompt length) --
--eval-batch-size controls how many responses go into one batch (default
20, i.e. the whole query at once; lower it if you hit memory limits with a
very large num_samples). Numerically identical results to the sequential
version (verified: padding with left-padding + attention_mask does not
change the logits produced for the real, unpadded tokens).

This module is deliberately independent of correctness labels and gold
answers (it never sees them), but it DOES require the generating model's
logits, which is why it cannot be fully decoupled from Stage A the way
S^(1)/S^(2) are.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.self_eval")

_DTYPE_MAP = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

_PROMPT_TEMPLATE = """{story}Question: {question}
Here are some brainstormed ideas: {few_shots}
Possible Answer: {answer}
Is the possible answer:
(A) True
(B) False
The possible answer is: ("""


def load_eval_model(
    model_name: str,
    device: str | None = None,
    quantize_4bit: bool = False,
    dtype: str = "bf16",
    gpu_id: int | None = None,
):
    """Load the SAME model used for Stage A generation for self-evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for batched next-token-logit extraction: it
    # guarantees the real last token of every sequence sits at position -1,
    # regardless of how much padding shorter prompts in the batch needed.
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    target = gpu_id if gpu_id is not None else 0

    if quantize_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--quantize-4bit requires a CUDA GPU.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, _DTYPE_MAP[dtype]),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": target}
        LOGGER.info("Loading in 4-bit NF4 (bitsandbytes) -- documented precision deviation.")
    else:
        resolved_dtype = getattr(torch, _DTYPE_MAP[dtype]) if torch.cuda.is_available() else torch.float32
        if dtype == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            LOGGER.warning("bf16 requested but not supported on this GPU; falling back to fp16.")
            resolved_dtype = torch.float16
        kwargs["torch_dtype"] = resolved_dtype
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": target}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    if device is not None:
        resolved_device = device
    elif torch.cuda.is_available():
        resolved_device = f"cuda:{target}"  # must match the device_map target above, or inputs land on the wrong GPU
    else:
        resolved_device = "cpu"
    if "device_map" not in kwargs:
        model.to(resolved_device)

    a_tok = tokenizer.encode("(A", add_special_tokens=False)[-1]
    b_tok = tokenizer.encode("(B", add_special_tokens=False)[-1]
    LOGGER.info("Loaded self-eval model=%s device=%s A_tok=%d B_tok=%d", model_name, resolved_device, a_tok, b_tok)
    return tokenizer, model, resolved_device, a_tok, b_tok


def _build_prompt(question: str, answer: str, other_answers: list[str], *, story: str, max_brainstorm: int) -> str:
    unique_others = sorted(set(a.lstrip() for a in other_answers))[:max_brainstorm]
    few_shots = "\n".join(unique_others)
    return _PROMPT_TEMPLATE.format(
        story=(story + "\n") if story else "",
        question=question,
        few_shots=few_shots,
        answer=answer.lstrip(),
    )


def p_true_for_responses_batch(
    question: str,
    answers: list[str],
    *,
    tokenizer,
    model,
    device: str,
    a_tok: int,
    b_tok: int,
    story: str = "",
    max_brainstorm: int = 10,
    batch_size: int = 20,
) -> list[float]:
    """Batched version of the official two-way-logit P(true) computation.

    Each response's prompt still uses that SAME response's own list of
    "other answers" as brainstormed ideas (matching the original,
    per-response logic exactly) -- only the tokenizer/model calls are
    batched, not the prompt content.
    """
    import torch

    prompts = []
    for i, ans in enumerate(answers):
        others = answers[:i] + answers[i + 1 :]
        prompts.append(_build_prompt(question, ans, others, story=story, max_brainstorm=max_brainstorm))

    p_true: list[float] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :]  # left-padded -> last real token is at -1 for every row

        pair_logits = logits[:, [a_tok, b_tok]].float()
        probs = torch.softmax(pair_logits, dim=-1)
        p_true.extend(probs[:, 0].detach().cpu().tolist())  # index 0 == "(A" == True

    return p_true


def build_self_eval_matrix(p_true: list[float]) -> np.ndarray:
    """v9 S^(3): diagonal = p_true_i, off-diagonal = geometric mean."""
    n = len(p_true)
    p = np.asarray(p_true, dtype=np.float64)
    matrix = np.sqrt(np.outer(p, p))
    np.fill_diagonal(matrix, p)
    return np.clip(matrix, 0.0, 1.0).astype(np.float32)


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
    model_name: str,
    limit: int | None,
    device: str | None,
    quantize_4bit: bool = False,
    dtype: str = "bf16",
    gpu_id: int | None = None,
    eval_batch_size: int = 20,
) -> None:
    tokenizer, model, resolved_device, a_tok, b_tok = load_eval_model(
        model_name, device=device, quantize_4bit=quantize_4bit, dtype=dtype, gpu_id=gpu_id
    )
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

            p_true = p_true_for_responses_batch(
                question, texts,
                tokenizer=tokenizer, model=model, device=resolved_device,
                a_tok=a_tok, b_tok=b_tok, batch_size=eval_batch_size,
            )

            matrix = build_self_eval_matrix(p_true)
            u_p_official = 1.0 - float(np.mean(p_true))

            result = {
                "id": query_id,
                "question": question,
                "responses": record["responses"],
                "source": {
                    "name": "self_eval",
                    "model": model_name,
                    "matrix_shape": list(matrix.shape),
                    "construction": "diagonal=p_true, off_diagonal=sqrt(p_true_i*p_true_j)",
                    "note": "U-ETF design choice; official MS-UQ baseline uses a scalar, see p_true_scalar_uncertainty",
                    "device": resolved_device,
                    "quantize_4bit": quantize_4bit,
                    "dtype": dtype,
                    "eval_batch_size": eval_batch_size,
                },
                "p_true": p_true,
                "p_true_scalar_uncertainty": u_p_official,
                "matrix": matrix.tolist(),
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            LOGGER.info("query=%s mean_p_true=%.4f U_p=%.4f", query_id, float(np.mean(p_true)), u_p_official)

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Stage-A JSONL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, help="MUST be the same model used for Stage A generation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--quantize-4bit", action="store_true", help="Kept for parity; not needed on H200.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gpu-id", type=int, default=None, help="Pin this run to one GPU (0-3 on a 4xH200 box)")
    parser.add_argument("--eval-batch-size", type=int, default=20, help="Responses per forward pass (default: whole query at once)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    process_file(
        args.input,
        args.output,
        model_name=args.model,
        limit=args.limit,
        device=args.device,
        quantize_4bit=args.quantize_4bit,
        dtype=args.dtype,
        gpu_id=args.gpu_id,
        eval_batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    main()
