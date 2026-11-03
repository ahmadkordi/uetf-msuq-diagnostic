#!/usr/bin/env python3
"""Stage A (TruthfulQA variant): Faithful MS-UQ-style generation for TruthfulQA. (H200-optimized)

This version loads TruthfulQA from a LOCAL DatasetDict saved with
save_to_disk() -- required for the offline server. Prepare the local copy
once on a machine with internet via:

    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation")
    ds.save_to_disk("/mnt/data/datasets/truthfulqa_local")

Then run with --truthfulqa-dir /mnt/data/datasets/truthfulqa_local

Phase 0 audit notes:
    TruthfulQA (Lin, Hilton & Evans, 2022) is an adversarial truthfulness
    benchmark: questions are designed so that common misconceptions
    reproduce high-consistency WRONG answers. This is the key benchmark for
    the v9 SSL-validity diagnostic (Experiment 0B): an SSL proxy that only
    rewards cross-sample agreement is expected to look confident here even
    when the model is factually wrong, and the paper must report this
    honestly.

    Like CoQA/NQ-Open/HotpotQA, generation is RAW COMPLETION, not
    chat-templated. TruthfulQA has no per-question context, so the prompt
    is a simple QA format:

        prompt = "Q: {question}\\nA:"

    No few-shot prefix by default (per the official MS-UQ TruthfulQA path,
    which uses the bare question). Stopping config follows the NQ-Open
    pattern (ban "Q:" multi-token sequence + EOS on ["\\n", ",", "."]).

Output JSONL: one line per question. "best_answer", "correct_answers",
"incorrect_answers", and "category" are kept ONLY for later Phase 8
correctness scoring -- Stage B onward never reads them (isolation rule
preserved).
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


def load_model(
    model_name: str,
    *,
    quantize_4bit: bool = False,
    dtype: str = "bf16",
    gpu_id: int | None = None,
):
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


def build_prompt(question: str, fs_block: str = "") -> str:
    """Raw-completion prompt. No context, no chat template.

    Format follows the plain QA style used by NQ-Open, since TruthfulQA
    questions are self-contained (no supporting passage).
    """
    if fs_block:
        return f"Answer these questions:{fs_block}\nQ: {question}\nA:"
    return f"Q: {question}\nA:"


def get_stopping_config(tokenizer) -> dict[str, Any]:
    """Tokenizer-agnostic stopping config (same pattern as NQ-Open / HotpotQA).

    Bans the exact multi-token sequence for "Q:" (prevents the model from
    hallucinating the next QA turn) and EOS on ["\\n", ",", "."].
    """
    cls_name = tokenizer.__class__.__name__

    def last_token(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer produced no tokens for {text!r}.")
        return ids[-1]

    if cls_name == "LlamaTokenizer":
        eos_token_id = [tokenizer(t)["input_ids"][-1] for t in ["\n", ",", "."]]
    elif cls_name == "GPT2Tokenizer":
        eos_token_id = [tokenizer.encode(t)[1] for t in ["\n", ",", "."]]
    else:
        LOGGER.warning(
            "Tokenizer class %s is not LlamaTokenizer/GPT2Tokenizer -- using "
            "tokenizer-agnostic fallback.",
            cls_name,
        )
        eos_token_id = [last_token(t) for t in ["\n", ",", "."]]
    eos_token_id += [tokenizer.eos_token_id]

    bad_words_ids = [tokenizer("Q:", add_special_tokens=False)["input_ids"]]
    return {"eos_token_id": eos_token_id, "bad_words_ids": bad_words_ids}


def _as_list(value: Any) -> list:
    """Normalize HF Dataset list fields (they can be np arrays or lists)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def load_truthfulqa_questions(
    truthfulqa_dir: str,
    num_questions: int | None = None,
    sample_seed: int = 20260903,
) -> list[dict[str, Any]]:
    """Load TruthfulQA (generation config) from a local DatasetDict."""
    from datasets import load_from_disk

    path = Path(truthfulqa_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Local TruthfulQA dataset not found at {path}. Prepare it on an "
            "internet-connected machine with "
            "load_dataset('truthful_qa', 'generation').save_to_disk(...)."
        )

    ds = load_from_disk(str(path))
    if "validation" in ds:
        data = ds["validation"]
    elif "train" in ds:
        data = ds["train"]
    else:
        first = list(ds.keys())[0]
        LOGGER.warning("No 'validation'/'train' split; using '%s' instead.", first)
        data = ds[first]

    indices = list(range(len(data)))
    if num_questions is not None and num_questions < len(indices):
        rng = random.Random(sample_seed)
        indices = sorted(rng.sample(indices, num_questions))
        LOGGER.info("Sampled %d/%d questions (seed=%d).", num_questions, len(data), sample_seed)

    records = []
    for idx in indices:
        item = data[idx]
        records.append(
            {
                "id": f"truthfulqa_{idx}",
                "question": item["question"],
                "best_answer": item.get("best_answer", ""),
                "correct_answers": _as_list(item.get("correct_answers", [])),
                "incorrect_answers": _as_list(item.get("incorrect_answers", [])),
                "category": item.get("category", ""),
            }
        )
    return records


def generate_samples(
    tokenizer,
    model,
    prompt: str,
    *,
    generation: GenerationConfig,
    stopping: dict[str, Any],
    q_seed: int,
) -> list[dict[str, Any]]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    if generation.batched_sampling:
        set_seed(q_seed)
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
        return [
            {
                "sample_id": sid,
                "text": tokenizer.decode(output[sid, prompt_len:], skip_special_tokens=True).strip(),
                "seed": q_seed,
            }
            for sid in range(generation.num_samples)
        ]

    responses = []
    for sample_id in range(generation.num_samples):
        sample_seed = q_seed + sample_id
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
    output_path: Path,
    generation: GenerationConfig,
    *,
    truthfulqa_dir: str,
    num_questions: int | None,
    sample_seed: int,
    limit: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(
        generation.model,
        quantize_4bit=generation.quantize_4bit,
        dtype=generation.dtype,
        gpu_id=generation.gpu_id,
    )
    LOGGER.info(
        "Loaded model: %s (4-bit=%s dtype=%s gpu_id=%s batched_sampling=%s)",
        generation.model, generation.quantize_4bit, generation.dtype,
        generation.gpu_id, generation.batched_sampling,
    )
    stopping = get_stopping_config(tokenizer)
    LOGGER.info("Stopping config: eos_token_id=%s bad_words_ids=%s",
                stopping["eos_token_id"], stopping["bad_words_ids"])

    records = load_truthfulqa_questions(truthfulqa_dir, num_questions=num_questions, sample_seed=sample_seed)
    LOGGER.info("Loaded %d TruthfulQA questions.", len(records))

    processed = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for rec in records:
            if limit is not None and processed >= limit:
                break

            prompt = build_prompt(rec["question"])
            q_seed = generation.seed + processed * 1000
            responses = generate_samples(
                tokenizer, model, prompt,
                generation=generation, stopping=stopping, q_seed=q_seed,
            )
            LOGGER.info("id=%s samples=%d/%d done", rec["id"], len(responses), generation.num_samples)

            result = {
                "id": rec["id"],
                "question": rec["question"],
                "responses": responses,
                "generation": asdict(generation),
                # ---- Phase 8 correctness fields (never read by Stage B) ----
                "best_answer": rec["best_answer"],
                "correct_answers": rec["correct_answers"],
                "incorrect_answers": rec["incorrect_answers"],
                "category": rec["category"],
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1

    LOGGER.info("Completed %d questions. Output: %s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--truthfulqa-dir",
        default=DEFAULT_TRUTHFULQA_DIR,
        help=f"Local path to the TruthfulQA DatasetDict saved with save_to_disk() "
             f"(default: {DEFAULT_TRUTHFULQA_DIR})",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N questions (quick tests)")
    parser.add_argument("--num-questions", type=int, default=None, help="Randomly subsample N questions")
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gpu-id", type=int, default=None, help="Pin this run to one GPU (0-3)")
    parser.add_argument("--no-batched-sampling", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

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
        args.output,
        generation,
        truthfulqa_dir=args.truthfulqa_dir,
        num_questions=args.num_questions,
        sample_seed=args.sample_seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()