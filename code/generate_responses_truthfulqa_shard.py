#!/usr/bin/env python3
"""Stage A (TruthfulQA) with shard support for multi-GPU parallelism."""

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

LOGGER = logging.getLogger("uetf.generate_truthfulqa")

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

DEFAULT_TRUTHFULQA_DIR = "/mnt/data/datasets/truthfulqa_local"


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    num_samples: int = 20
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 256
    seed: int = 20260903
    quantize_4bit: bool = False
    dtype: str = "bf16"
    batched_sampling: bool = True
    gpu_id: int | None = None


def set_seed(seed: int, strict: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if strict:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def load_model(model_name, *, quantize_4bit=False, dtype="bf16", gpu_id=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    target = gpu_id if gpu_id is not None else 0
    kwargs = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": target}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=_DTYPE_MAP[dtype],
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    else:
        resolved_dtype = _DTYPE_MAP[dtype] if torch.cuda.is_available() else torch.float32
        kwargs["torch_dtype"] = resolved_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def build_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def get_stopping_config(tokenizer):
    cls_name = tokenizer.__class__.__name__

    def first_token(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"No tokens for {text!r}")
        return ids[0]

    if cls_name == "LlamaTokenizer":
        eos_token_id = [tokenizer.encode(t)[-1] for t in [".", "\n"]] + [29889]
    elif cls_name == "GPT2Tokenizer":
        eos_token_id = [first_token(t) for t in [".", "\n"]]
    else:
        if tokenizer.eos_token_id:
            eos_token_id = [first_token(t) for t in [".", "\n"]] + [tokenizer.eos_token_id]
        else:
            eos_token_id = [first_token(t) for t in [".", "\n"]]
    eos_token_id += [tokenizer.eos_token_id]

    question_framing = ["Question:", " Question:", "\n", "Answer:", " Answer:", "Q:"]
    bad_words_ids = [[first_token(t)] for t in question_framing]
    return {"eos_token_id": eos_token_id, "bad_words_ids": bad_words_ids}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def load_truthfulqa_questions(truthfulqa_dir, *, num_questions=None, sample_seed=20260903,
                              shard_id=0, num_shards=1):
    from datasets import load_from_disk

    ds = load_from_disk(truthfulqa_dir)
    if "validation" in ds:
        data = ds["validation"]
    elif "train" in ds:
        data = ds["train"]
    else:
        data = ds[list(ds.keys())[0]]

    indices = list(range(len(data)))
    if num_questions is not None and num_questions < len(indices):
        rng = random.Random(sample_seed)
        indices = sorted(rng.sample(indices, num_questions))

    # Shard filter
    indices = [idx for i, idx in enumerate(indices) if i % num_shards == shard_id]

    records = []
    for idx in indices:
        item = data[idx]
        records.append({
            "id": f"truthfulqa_{idx}",
            "question": item["question"],
            "best_answer": item.get("best_answer", ""),
            "correct_answers": _as_list(item.get("correct_answers", [])),
            "incorrect_answers": _as_list(item.get("incorrect_answers", [])),
            "category": item.get("category", ""),
        })
    return records


def generate_samples(tokenizer, model, prompt, *, generation, stopping, q_seed):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    if generation.batched_sampling:
        set_seed(q_seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=True,
                temperature=generation.temperature, top_p=generation.top_p,
                top_k=generation.top_k if generation.top_k > 0 else None,
                max_new_tokens=generation.max_new_tokens,
                num_return_sequences=generation.num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stopping["eos_token_id"],
                bad_words_ids=stopping["bad_words_ids"])
        return [
            {"sample_id": sid,
             "text": tokenizer.decode(output[sid, prompt_len:], skip_special_tokens=True).strip(),
             "seed": q_seed}
            for sid in range(generation.num_samples)
        ]

    responses = []
    for sample_id in range(generation.num_samples):
        sample_seed = q_seed + sample_id
        set_seed(sample_seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=True,
                temperature=generation.temperature, top_p=generation.top_p,
                top_k=generation.top_k if generation.top_k > 0 else None,
                max_new_tokens=generation.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stopping["eos_token_id"],
                bad_words_ids=stopping["bad_words_ids"])
        text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
        responses.append({"sample_id": sample_id, "text": text, "seed": sample_seed})
    return responses


def generate_dataset(output_path, generation, *, truthfulqa_dir, num_questions,
                     sample_seed, limit, shard_id, num_shards):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(
        generation.model, quantize_4bit=generation.quantize_4bit,
        dtype=generation.dtype, gpu_id=generation.gpu_id)
    LOGGER.info("Loaded model=%s gpu=%s dtype=%s shard=%d/%d",
                generation.model, generation.gpu_id, generation.dtype,
                shard_id, num_shards)
    stopping = get_stopping_config(tokenizer)
    LOGGER.info("Stopping config: eos=%s bad_words=%s",
                stopping["eos_token_id"], stopping["bad_words_ids"])

    records = load_truthfulqa_questions(
        truthfulqa_dir, num_questions=num_questions,
        sample_seed=sample_seed, shard_id=shard_id, num_shards=num_shards)
    LOGGER.info("Shard %d/%d: loaded %d records", shard_id, num_shards, len(records))

    processed = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for rec in records:
            if limit is not None and processed >= limit:
                break
            prompt = build_prompt(rec["question"])
            q_seed = generation.seed + processed * 1000
            responses = generate_samples(tokenizer, model, prompt,
                                          generation=generation, stopping=stopping,
                                          q_seed=q_seed)
            result = {
                "id": rec["id"],
                "question": rec["question"],
                "responses": responses,
                "generation": asdict(generation),
                "best_answer": rec["best_answer"],
                "correct_answers": rec["correct_answers"],
                "incorrect_answers": rec["incorrect_answers"],
                "category": rec["category"],
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            if processed % 50 == 0:
                LOGGER.info("[shard %d/%d] processed=%d/%d",
                            shard_id, num_shards, processed, len(records))

    LOGGER.info("Shard %d/%d DONE. processed=%d output=%s",
                shard_id, num_shards, processed, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--truthfulqa-dir", default=DEFAULT_TRUTHFULQA_DIR)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-batched-sampling", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    generation = GenerationConfig(
        model=args.model, num_samples=args.num_samples,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        quantize_4bit=args.quantize_4bit, dtype=args.dtype,
        batched_sampling=not args.no_batched_sampling, gpu_id=args.gpu_id)
    generate_dataset(args.output, generation,
                     truthfulqa_dir=args.truthfulqa_dir,
                     num_questions=args.num_questions,
                     sample_seed=args.sample_seed,
                     limit=args.limit,
                     shard_id=args.shard_id, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
