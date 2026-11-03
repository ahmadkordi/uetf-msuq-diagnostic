#!/usr/bin/env python3
"""Stage A (CoQA variant): Faithful MS-UQ-style generation for CoQA. (H200-optimized)

Phase 0 audit note (github.com/tiejin98/MSUQ, dataeval/coqa.py):
    Unlike our 25-query pilot (which used tokenizer.apply_chat_template),
    the official CoQA generation path is RAW COMPLETION, not chat-templated:
        prompt = story_so_far + " Q: " + question + " A:"
    where `story_so_far` starts as the CoQA passage and, after every turn,
    has the GOLD (not model-generated) question/answer appended. Generation
    also uses custom `bad_words_ids`/`eos_token_id` so the model stops after
    one answer instead of hallucinating the next turn itself. Both
    reproduced exactly.

    Per the user's explicit priority (full baseline fidelity), this script
    defaults to the OFFICIAL sampling settings (T=1.0, top_p=1.0, top_k=0)
    rather than the 25-query pilot's controlled setting -- override via CLI
    if you want the pilot's setting instead.

H200 optimization notes (this revision -- see generate_responses_seed_v3.py
for the fuller rationale, mirrored here):
    - --quantize-4bit kept (opt-in) but no longer the default path; bf16 is
      now the default dtype (Hopper-native, more stable than fp16).
    - BATCHED SAMPLING: N=20 samples/turn generated in one model.generate()
      call via num_return_sequences=N (bad_words_ids/eos_token_id apply
      per-sequence, unaffected by batching), seeded once per turn instead of
      once per sample. --no-batched-sampling restores the old per-sample-seed
      behavior.
    - --gpu-id N pins the run to one specific GPU for easy 4-way parallelism.

Input: the official CoQA dev file, downloaded from
    https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
Output JSONL: one line per (conversation, turn); gold_answer/additional_answers
    are kept ONLY for later Phase 8 correctness scoring -- Stage B onward
    never reads them (isolation rule preserved).
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


def load_model(model_name: str, *, quantize_4bit: bool = False, dtype: str = "bf16", gpu_id: int | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = gpu_id if gpu_id is not None else 0
    if quantize_4bit and torch.cuda.is_available():
        device_map: str | dict | None = {"": target}
    elif torch.cuda.is_available():
        device_map = {"": target} if gpu_id is not None else "auto"
    else:
        device_map = None
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if device_map is not None:
        kwargs["device_map"] = device_map

    if quantize_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--quantize-4bit requires a CUDA GPU.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_DTYPE_MAP[dtype],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        LOGGER.info("Loading in 4-bit NF4 (bitsandbytes) -- documented precision deviation.")
    else:
        resolved_dtype = _DTYPE_MAP[dtype] if torch.cuda.is_available() else torch.float32
        if dtype == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            LOGGER.warning("bf16 requested but not supported on this GPU; falling back to fp16.")
            resolved_dtype = torch.float16
        kwargs["torch_dtype"] = resolved_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def get_stopping_config(tokenizer) -> dict[str, Any]:
    """Faithful, tokenizer-agnostic port of dataeval/coqa.py::_generate_config.

    The official code hardcodes index [1] into `tokenizer.encode(...)`,
    assuming a BOS token is always prepended -- true for LlamaTokenizer's
    SentencePiece encoding, but NOT for Qwen2's fast BPE tokenizer. We
    encode with add_special_tokens=False and take index [0] instead, which
    is equivalent on BOS-prepending tokenizers and simply correct otherwise.
    """
    cls_name = tokenizer.__class__.__name__

    def first_token(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer produced no tokens for {text!r}.")
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


def load_coqa_conversations(
    coqa_path: Path,
    *,
    num_conversations: int | None = None,
    conv_seed: int = 20260903,
) -> list[dict[str, Any]]:
    """Port of dataeval/coqa.py::_save_dataset -- builds per-turn records
    with the cumulative, gold-answer-injected story field.

    If num_conversations is given, randomly samples that many whole
    conversations (all their turns) with a fixed seed -- subsampling must
    keep whole conversations together since each turn's cumulative context
    depends on earlier turns WITHIN its own conversation.
    """
    with coqa_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)["data"]

    if num_conversations is not None and num_conversations < len(data):
        rng = random.Random(conv_seed)
        original_count = len(data)
        data = rng.sample(data, num_conversations)
        LOGGER.info("Sampled %d/%d conversations (seed=%d).", num_conversations, original_count, conv_seed)

    records: list[dict[str, Any]] = []
    for sample in data:
        story = sample["story"]
        questions = sample["questions"]
        answers = sample["answers"]
        additional_answers = sample.get("additional_answers", {})
        for turn_index, question in enumerate(questions):
            gold_answer = answers[turn_index]["input_text"]
            extra = [
                additional_answers[str(i)][turn_index]["input_text"]
                for i in range(3)
                if str(i) in additional_answers
            ]
            records.append(
                {
                    "id": f"{sample['id']}_{turn_index}",
                    "conversation_id": sample["id"],
                    "turn_index": turn_index,
                    "story_so_far": story,
                    "question": question["input_text"],
                    "gold_answer": gold_answer,
                    "additional_answers": extra,
                }
            )
            story = story + " Q: " + question["input_text"] + " A: " + gold_answer
            if not story.endswith("."):
                story = story + "."
    return records


def generate_samples(
    tokenizer,
    model,
    prompt: str,
    *,
    generation: GenerationConfig,
    stopping: dict[str, Any],
    turn_seed: int,
) -> list[dict[str, Any]]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    if generation.batched_sampling:
        set_seed(turn_seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_k=generation.top_k if generation.top_k > 0 else None,
                max_new_tokens=generation.max_new_tokens,
                num_return_sequences=generation.num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stopping["eos_token_id"],
                bad_words_ids=stopping["bad_words_ids"],
            )
        responses = []
        for sample_id in range(generation.num_samples):
            text = tokenizer.decode(output[sample_id, prompt_len:], skip_special_tokens=True).strip()
            responses.append({"sample_id": sample_id, "text": text, "seed": turn_seed})
        return responses

    responses = []
    for sample_id in range(generation.num_samples):
        sample_seed = turn_seed + sample_id
        set_seed(sample_seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_k=generation.top_k if generation.top_k > 0 else None,
                max_new_tokens=generation.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stopping["eos_token_id"],
                bad_words_ids=stopping["bad_words_ids"],
            )
        text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
        responses.append({"sample_id": sample_id, "text": text, "seed": sample_seed})
    return responses


def generate_dataset(
    coqa_path: Path,
    output_path: Path,
    generation: GenerationConfig,
    limit: int | None = None,
    num_conversations: int | None = None,
    conv_seed: int = 20260903,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(
        generation.model, quantize_4bit=generation.quantize_4bit, dtype=generation.dtype, gpu_id=generation.gpu_id
    )
    LOGGER.info(
        "Loaded model: %s (4-bit=%s dtype=%s gpu_id=%s batched_sampling=%s)",
        generation.model, generation.quantize_4bit, generation.dtype, generation.gpu_id, generation.batched_sampling,
    )
    stopping = get_stopping_config(tokenizer)
    LOGGER.info("Stopping config: eos_token_id=%s bad_words_ids=%s", stopping["eos_token_id"], stopping["bad_words_ids"])

    records = load_coqa_conversations(coqa_path, num_conversations=num_conversations, conv_seed=conv_seed)
    LOGGER.info("Loaded %d (conversation, turn) records from %s", len(records), coqa_path)

    processed = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for rec in records:
            if limit is not None and processed >= limit:
                break

            prompt = rec["story_so_far"] + " Q: " + rec["question"] + " A:"
            turn_seed = generation.seed + processed * 1000
            responses = generate_samples(
                tokenizer, model, prompt, generation=generation, stopping=stopping, turn_seed=turn_seed
            )
            LOGGER.info("id=%s samples=%d/%d done", rec["id"], len(responses), generation.num_samples)

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

    LOGGER.info("Completed %d (conversation, turn) records. Output: %s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--coqa-file", required=True, type=Path, help="Path to coqa-dev-v1.0.json")
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
    parser.add_argument("--quantize-4bit", action="store_true", help="Kept for parity; not needed on H200.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gpu-id", type=int, default=None, help="Pin this run to one GPU (0-3 on a 4xH200 box)")
    parser.add_argument("--no-batched-sampling", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    generation = GenerationConfig(
        model=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        quantize_4bit=args.quantize_4bit,
        dtype=args.dtype,
        batched_sampling=not args.no_batched_sampling,
        gpu_id=args.gpu_id,
    )
    generate_dataset(
        args.coqa_file, args.output, generation,
        limit=args.limit, num_conversations=args.num_conversations, conv_seed=args.conv_seed,
    )


if __name__ == "__main__":
    main()
