import sys
import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_module(path):
    spec = importlib.util.spec_from_file_location("hotpot_stage_a", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)

    # Required so dataclasses/type inspection can resolve __module__
    # correctly during exec_module() on Python 3.12.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module

def load_records(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for resp in r["responses"]:
                if not resp.get("text", "").strip():
                    rows.append((r, resp))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", required=True)
    ap.add_argument("--hotpot-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=20)
    args = ap.parse_args()

    stage_a = Path(args.stage_a)
    hotpot = load_module("/mnt/src/generate_responses_hotpotqa.py")

    empty_cases = load_records(stage_a)
    print(f"EMPTY CASES: {len(empty_cases)}")
    print(f"ANALYZING FIRST: {min(args.max_cases, len(empty_cases))}")

    ds = load_from_disk(args.hotpot_dir)
    split = ds["validation"]

    qmap = {}
    for item in split:
        qid = str(item.get("id", item.get("_id", "")))
        qmap[qid] = item

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(f"cuda:{args.gpu_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.gpu_id},
        low_cpu_mem_usage=True,
    )
    model.eval()

    rng = random.Random(20260920)
    cases = rng.sample(empty_cases, min(args.max_cases, len(empty_cases)))

    for n, (record, resp) in enumerate(cases, 1):
        qid = record["id"]
        sample_id = resp["sample_id"]
        seed = resp["seed"]

        item = qmap.get(str(qid))
        if item is None:
            print(f"\n[{n}] {qid} -> DATASET RECORD NOT FOUND")
            continue

        prompt = hotpot.build_prompt(
            record["question"],
            item["context"],
        )

        stopping = hotpot.get_stopping_config(tokenizer)

        # Reproduce the SAME sampling seed/config but request only 1 token.
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=None,
                max_new_tokens=1,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stopping["eos_token_id"],
                bad_words_ids=stopping["bad_words_ids"],
            )

        new_ids = out[0, prompt_len:]
        token_id = int(new_ids[0]) if len(new_ids) else None
        token_text = (
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if token_id is not None
            else ""
        )

        is_eos = token_id in stopping["eos_token_id"] if token_id is not None else False

        print(f"\n[{n}]")
        print(f"qid        = {qid}")
        print(f"sample_id  = {sample_id}")
        print(f"seed       = {seed}")
        print(f"question   = {record['question']}")
        print(f"token_id   = {token_id}")
        print(f"token_repr = {token_text!r}")
        print(f"is_eos     = {is_eos}")
        print(f"eos_ids    = {stopping['eos_token_id']}")

        if is_eos:
            print("DIAGNOSIS  = EMPTY CAUSED BY STOP/EOS TOKEN")
        else:
            print("DIAGNOSIS  = NOT EXPLAINED BY EOS; investigate decoding/bad-word behavior")


if __name__ == "__main__":
    main()
