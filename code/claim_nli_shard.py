#!/usr/bin/env python3
"""Stage B.3: Build the claim-level relation matrix S^(2) for U-ETF. (H200-optimized)

Shard support: --shard-id and --num-shards split the input across processes
so multiple GPUs can work in parallel. Each shard writes its own output;
merge with `cat` at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from semantic_nli import build_semantic_nli_matrix, load_nli_model

LOGGER = logging.getLogger("uetf.claim_nli")


def validate_claims_record(record: dict[str, Any], line_no: int) -> tuple[str, str, list[str]]:
    if "id" not in record or "question" not in record:
        raise ValueError(f"Input line {line_no} must contain 'id' and 'question'.")
    claims_block = record.get("claims")
    if not isinstance(claims_block, dict) or "per_response" not in claims_block:
        raise ValueError(f"Input line {line_no} must contain 'claims.per_response' (see extract_claims.py).")

    per_response = claims_block["per_response"]
    if not per_response:
        raise ValueError(f"Input line {line_no} has an empty claims list.")

    texts: list[str] = []
    for idx, item in enumerate(per_response):
        text = str(item.get("claims_text", "")).strip()
        if not text:
            LOGGER.warning(
                "id=%s response=%d has an empty claims_text; using empty string.",
                record["id"],
                idx,
            )
        texts.append(text)

    return str(record["id"]), str(record["question"]), texts


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

            # Shard filter
            if num_shards > 1 and (line_no - 1) % num_shards != shard_id:
                skipped += 1
                continue

            if limit is not None and processed >= limit:
                break

            record = json.loads(line)
            query_id, question, claim_texts = validate_claims_record(record, line_no)

            matrices = build_semantic_nli_matrix(
                question,
                claim_texts,
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
                    "name": "claim_nli",
                    "model": model_name,
                    "matrix_shape": list(matrices["agreement"].shape),
                    "symmetrization": "bidirectional_mean" if symmetrize else "raw_directed",
                    "question_conditioned": True,
                    "input_text": "claims_text (see extract_claims.py)",
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
            if processed % 100 == 0:
                LOGGER.info("[shard %d/%d] processed=%d skipped=%d",
                            shard_id, num_shards, processed, skipped)

    LOGGER.info("Shard %d/%d done. processed=%d skipped=%d output=%s",
                shard_id, num_shards, processed, skipped, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-symmetrize", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
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