#!/usr/bin/env python3
"""Stage B.3: Build the claim-level relation matrix S^(2) for U-ETF. (H200-optimized)

Phase 0 audit note (github.com/tiejin98/MSUQ, getting_claim_pkl.py + sim_mat.py):
    The official "knowledge" (D-UE) matrix is NOT a separate algorithm --
    it replaces each response's raw text with that response's joined-claims
    string (from extract_claims.py) and then runs the EXACT SAME NLI
    machinery used for S^(1). So this module is a thin wrapper around
    semantic_nli.build_semantic_nli_matrix.

H200 optimization note: same rationale as semantic_nli.py -- default
--batch-size raised to 128, optional --dtype, --gpu-id for 4-way
parallelism. No algorithmic change.

This module is deliberately independent of correctness labels, gold answers,
and the U-ETF reliability/gating layer.

Input:  output of extract_claims.py (id, question, responses, claims.per_response)
Output: same schema as semantic_nli.py's Stage-B artifact, source.name="claim_nli"
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
                "id=%s response=%d has an empty claims_text (extraction failed or "
                "unwanted answer); using empty string as its claim content.",
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
) -> None:
    tokenizer, model, resolved_device, entailment_id, contradiction_id = load_nli_model(
        model_name, device=device, dtype=dtype, gpu_id=gpu_id
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
                },
                "matrix_agreement": matrices["agreement"].tolist(),
                "matrix_disagreement": matrices["disagreement"].tolist(),
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            LOGGER.info("query=%s matrix_shape=%s", query_id, matrices["agreement"].shape)

    LOGGER.info("Completed %d queries. Output=%s", processed, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Output of extract_claims.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--gpu-id", type=int, default=None, help="Pin this run to one GPU (0-3 on a 4xH200 box)")
    parser.add_argument("--no-symmetrize", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
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
    )


if __name__ == "__main__":
    main()
