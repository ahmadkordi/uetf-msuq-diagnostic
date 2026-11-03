#!/usr/bin/env python3
"""Stage B: Build the semantic NLI relation matrix S^(1) for U-ETF. (H200-optimized)

Shard support: --shard-id and --num-shards split the input file across
processes so multiple GPUs can work in parallel on the same dataset.
Each shard writes to its own output; merge with `cat` at the end.

Empty-text policy: responses with empty text are no longer a hard error;
they are logged with a warning and kept as empty strings (NLI treats them
as uninformative statements). This matches the documented ~1.9% empty rate
from the official MS-UQ stopping config (`.`/`,`/`\n`/EOS + T=1.0).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("uetf.semantic_nli")
DEFAULT_MODEL = "microsoft/deberta-large-mnli"
_DTYPE_MAP = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def load_nli_model(model_name: str, device: str | None = None, dtype: str = "fp32", gpu_id: int | None = None):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "deberta" and dtype != "fp32":
        LOGGER.warning(
            "%s is legacy DeBERTa-v1 -- bf16/fp16 crashes in its attention "
            "implementation. Forcing fp32.",
            model_name,
        )
        dtype = "fp32"

    resolved_dtype = getattr(torch, _DTYPE_MAP[dtype])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=resolved_dtype)
    model.eval()

    if device is None:
        if torch.cuda.is_available():
            device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
        else:
            device = "cpu"
    model.to(device)

    labels = {int(k): str(v).strip().lower() for k, v in model.config.id2label.items()}
    entailment_ids = [idx for idx, label in labels.items() if "entail" in label]
    contradiction_ids = [idx for idx, label in labels.items() if "contra" in label]
    if len(entailment_ids) != 1:
        raise ValueError(f"Could not uniquely identify entailment label in {labels}.")
    if len(contradiction_ids) != 1:
        raise ValueError(f"Could not uniquely identify contradiction label in {labels}.")

    entailment_id = entailment_ids[0]
    contradiction_id = contradiction_ids[0]
    LOGGER.info(
        "Loaded NLI model=%s device=%s dtype=%s entailment_id=%d contradiction_id=%d",
        model_name, device, dtype, entailment_id, contradiction_id,
    )
    return tokenizer, model, device, entailment_id, contradiction_id


def validate_responses(record: dict[str, Any], line_no: int) -> tuple[str, str, list[str]]:
    """Validate one Stage-A artifact. Empty texts produce a warning, not an error."""
    if "id" not in record or "question" not in record:
        raise ValueError(f"Input line {line_no} must contain 'id' and 'question'.")

    responses = record.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"Input line {line_no} must contain a non-empty 'responses' list.")

    texts: list[str] = []
    for idx, item in enumerate(responses):
        if not isinstance(item, dict) or "text" not in item:
            raise ValueError(f"Input line {line_no}, response {idx} must contain 'text'.")
        text = str(item["text"]).strip()
        if not text:
            LOGGER.warning(
                "id=%s response=%d has empty text; using empty string "
                "(downstream NLI will treat it as an uninformative statement).",
                record.get("id", "?"), idx,
            )
        texts.append(text)

    return str(record["id"]), str(record["question"]), texts


def nli_probabilities(
    premises: list[str],
    hypotheses: list[str],
    *,
    tokenizer,
    model,
    device: str,
    entailment_id: int,
    contradiction_id: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if len(premises) != len(hypotheses):
        raise ValueError("premises and hypotheses must have equal length.")

    import torch

    entail_values: list[np.ndarray] = []
    contra_values: list[np.ndarray] = []
    for start in range(0, len(premises), batch_size):
        p_batch = premises[start : start + batch_size]
        h_batch = hypotheses[start : start + batch_size]
        joined = [f"{p} [SEP] {h}" for p, h in zip(p_batch, h_batch)]

        encoded = tokenizer(
            joined,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=-1)
        entail_values.append(probs[:, entailment_id].detach().cpu().numpy())
        contra_values.append(probs[:, contradiction_id].detach().cpu().numpy())

    return (
        np.concatenate(entail_values).astype(np.float32),
        np.concatenate(contra_values).astype(np.float32),
    )


def build_semantic_nli_matrix(
    question: str,
    responses: list[str],
    *,
    tokenizer,
    model,
    device: str,
    entailment_id: int,
    contradiction_id: int,
    batch_size: int = 128,
    symmetrize: bool = True,
) -> dict[str, np.ndarray]:
    n = len(responses)
    if n == 0:
        raise ValueError("responses must not be empty.")

    premises: list[str] = []
    hypotheses: list[str] = []
    pair_index: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            premises.append(f"{question} {responses[i]}")
            hypotheses.append(f"{question} {responses[j]}")
            pair_index.append((i, j))

    entail_flat, contra_flat = nli_probabilities(
        premises,
        hypotheses,
        tokenizer=tokenizer,
        model=model,
        device=device,
        entailment_id=entailment_id,
        contradiction_id=contradiction_id,
        batch_size=batch_size,
    )

    entail_directed = np.zeros((n, n), dtype=np.float32)
    contra_directed = np.zeros((n, n), dtype=np.float32)
    for (i, j), p_e, p_c in zip(pair_index, entail_flat, contra_flat):
        entail_directed[i, j] = p_e
        contra_directed[i, j] = p_c
    np.fill_diagonal(entail_directed, 1.0)
    np.fill_diagonal(contra_directed, 0.0)

    if symmetrize:
        entail_final = 0.5 * (entail_directed + entail_directed.T)
        contra_final = 0.5 * (contra_directed + contra_directed.T)
    else:
        entail_final = entail_directed
        contra_final = contra_directed

    agreement = np.clip(entail_final, 0.0, 1.0).astype(np.float32)
    disagreement = np.clip(1.0 - contra_final, 0.0, 1.0).astype(np.float32)
    return {"agreement": agreement, "disagreement": disagreement}


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str,
    batch_size: int,
    limit: int | None,
    device: str | None,
    symmetrize: bool,
    dtype: str = "fp32",
    gpu_id: int | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> None:
    tokenizer, model, resolved_device, entailment_id, contradiction_id = load_nli_model(
        model_name, device=device, dtype=dtype, gpu_id=gpu_id
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue

            # --- Shard filtering ---
            if num_shards > 1 and (line_no - 1) % num_shards != shard_id:
                skipped += 1
                continue

            if limit is not None and processed >= limit:
                break

            record = json.loads(line)
            query_id, question, responses = validate_responses(record, line_no)
            matrices = build_semantic_nli_matrix(
                question,
                responses,
                tokenizer=tokenizer,
                model=model,
                device=resolved_device,
                entailment_id=entailment_id,
                contradiction_id=contradiction_id,
                batch_size=batch_size,
                symmetrize=symmetrize,
            )

            result = {
                "id": query_id,
                "question": question,
                "responses": record["responses"],
                "source": {
                    "name": "semantic_nli",
                    "model": model_name,
                    "matrix_shape": list(matrices["agreement"].shape),
                    "symmetrization": "bidirectional_mean" if symmetrize else "raw_directed",
                    "question_conditioned": True,
                    "device": resolved_device,
                    "dtype": dtype,
                    "batch_size": batch_size,
                    "entailment_label_id": entailment_id,
                    "contradiction_label_id": contradiction_id,
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                },
                "matrix_agreement": matrices["agreement"].tolist(),
                "matrix_disagreement": matrices["disagreement"].tolist(),
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            if processed % 50 == 0:
                LOGGER.info("[shard %d/%d] processed=%d skipped=%d last_query=%s",
                            shard_id, num_shards, processed, skipped, query_id)

    LOGGER.info("Shard %d/%d done. processed=%d skipped=%d Output=%s",
                shard_id, num_shards, processed, skipped, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-symmetrize", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Which shard this process handles (0-indexed).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards; lines assigned by (line_no-1) %% num_shards.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard-id must be in [0, {args.num_shards - 1}].")

    process_file(
        args.input,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        symmetrize=not args.no_symmetrize,
        dtype=args.dtype,
        gpu_id=args.gpu_id,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()