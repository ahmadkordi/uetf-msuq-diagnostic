#!/usr/bin/env python3
"""Step 1 of the Llama-2 P(true) token test (tokenizer only, no GPU, no model weights needed).
Prints how each generator's tokenizer encodes the answer options used by the two-way softmax: "(A", "(B", " (A", "A", "B", " A", " B",
and whether the FIRST token of "(A" and "(B" is the same id (if it is, a softmax over first-token logits is constant 0.5 whatever the model thinks).
Usage: python token_test.py <model_dir_llama2> [<model_dir_llama31> <model_dir_qwen>]"""
import sys
from transformers import AutoTokenizer
for path in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    print("=" * 70); print(path, type(tok).__name__)
    for s in ["(A", "(B", " (A", " (B", "A", "B", " A", " B", "\n(A", "\n(B", "Answer: (A", "Answer: (B"]:
        ids = tok.encode(s, add_special_tokens=False); print(f"{s!r:16} ids={ids} pieces={tok.convert_ids_to_tokens(ids)}")
    a, b = tok.encode("(A", add_special_tokens=False), tok.encode("(B", add_special_tokens=False)
    print("first token of '(A' == first token of '(B' ?", a[0] == b[0], "| last token ids:", a[-1], b[-1])
    # encode in context: the prompt ends with the option letter position; check the tokens AFTER a fixed prefix
    pre = "Possible answers:\n(A) True\n(B) False\nThe answer is:"
    ctx = tok.encode(pre, add_special_tokens=False)
    for opt in ["(A", " (A", "(B", " (B"]:
        full = tok.encode(pre + opt, add_special_tokens=False)
        print(f"prefix+{opt!r:6} -> new tokens:", tok.convert_ids_to_tokens(full[len(ctx):]) if full[:len(ctx)] == ctx else "prefix tokenisation changes!")
