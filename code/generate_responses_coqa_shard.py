#!/usr/bin/env python3
"""Stage A (CoQA) with shard support for multi-GPU parallelism."""

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

LOGGER = logging.getLogger("uetf.generate_coqa")

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


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


def load_coqa_conversations(coqa_path, *, num_conversations=None, conv_seed=20260903):
    with coqa_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)["data"]
    if num_conversations is not None and num_conversations < len(data):
        rng = random.Random(conv_seed)
        original_count = len(data)
        data = rng.sample(data, num_conversations)
        LOGGER.info("Sampled %d/%d conversations.", num_conversations, original_count)

    records = []
    for sample in data:
        story = sample["story"]
        questions = sample["questions"]
        answers = sample["answers"]
        additional_answers = sample.get("additional_answers", {})
        for turn_index, question in enumerate(questions):
            gold_answer = answers[turn_index]["input_text"]
            extra = [additional_answers[str(i)][turn_index]["input_text"]
                     for i in range(3) if str(i) in additional_answers]
            records.append({
                "id": f"{sample['id']}_{turn_index}",
                "conversation_id": sample["id"],
                "turn_index": turn_index,
                "story_so_far": story,
                "question": question["input_text"],
                "gold_answer": gold_answer,
                "additional_answers": extra,
            })
            story = story + " Q: " + question["input_text"] + " A: " + gold_answer
            if not story.endswith("."):
                story = story + "."
    return records


def generate_samples(tokenizer, model, prompt, *, generation, stopping, turn_seed):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    if generation.batched_sampling:
        set_seed(turn_seed)
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
             "seed": turn_seed}
            for sid in range(generation.num_samples)
        ]

    responses = []
    for sample_id in range(generation.num_samples):
        sample_seed = turn_seed + sample_id
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


def generate_dataset(coqa_path, output_path, generation, *, limit=None,
                     num_conversations=None, conv_seed=20260903,
                     shard_id=0, num_shards=1):
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

    records = load_coqa_conversations(coqa_path, num_conversations=num_conversations,
                                       conv_seed=conv_seed)
    LOGGER.info("Loaded %d records", len(records))

    if num_shards > 1:
        before = len(records)
        records = [r for i, r in enumerate(records) if i % num_shards == shard_id]
        LOGGER.info("Shard %d/%d: %d/%d assigned", shard_id, num_shards,
                    len(records), before)

    processed = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for rec in records:
            if limit is not None and processed >= limit:
                break
            prompt = rec["story_so_far"] + " Q: " + rec["question"] + " A:"
            turn_seed = generation.seed + processed * 1000
            responses = generate_samples(tokenizer, model, prompt,
                                          generation=generation, stopping=stopping,
                                          turn_seed=turn_seed)
            result = {
                "id": rec["id"],
                "conversation_id": rec["conversation_id"],
                "turn_index": rec["turn_index"],
                "question": rec["question"],
                "story_so_far": rec["story_so_far"],
                "responses": responses,
                "generation": asdict(generation),
                "gold_answer": rec["gold_answer"],
                "additional_answers": rec["additional_answers"],
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            if processed % 100 == 0:
                LOGGER.info("[shard %d/%d] processed=%d", shard_id, num_shards, processed)

    LOGGER.info("Shard %d/%d DONE. processed=%d output=%s",
                shard_id, num_shards, processed, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--coqa-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-conversations", type=int, default=None)
    parser.add_argument("--conv-seed", type=int, default=20260903)
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
    generate_dataset(args.coqa_file, args.output, generation,
                     limit=args.limit, num_conversations=args.num_conversations,
                     conv_seed=args.conv_seed,
                     shard_id=args.shard_id, num_shards=args.num_shards)


if __name__ == "__main__":
    main()