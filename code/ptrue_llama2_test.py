import sys, json, math, torch, inspect
import self_eval as se

model_path = "/mnt/data/models/Llama-2-13b-hf"
stage_a = sys.argv[1]            # /mnt/output/stage_a_coqa_Llama2_13b.jsonl
NQ = 20

rows = [json.loads(l) for l in open(stage_a)][:NQ]
print("KEYS:", list(rows[0].keys()))
qk = next(k for k in rows[0] if "question" in k.lower())
ak = next(k for k in rows[0] if isinstance(rows[0][k], list) and rows[0][k]
          and (isinstance(rows[0][k][0], str) or isinstance(rows[0][k][0], dict)))
def txt(x): return x if isinstance(x, str) else next(v for v in x.values() if isinstance(v, str))
sk = next((k for k in rows[0] if k.lower() in ("story", "context")), None)
print("question key:", qk, "| answers key:", ak, "| story key:", sk)

print(inspect.signature(se.load_eval_model))
tok, model, dev, a_tok, b_tok = se.load_eval_model(model_path)
print("a_tok", a_tok, repr(tok.convert_ids_to_tokens(a_tok)), "b_tok", b_tok, repr(tok.convert_ids_to_tokens(b_tok)))
par = tok.encode(" (", add_special_tokens=False); print("' (' ->", par, tok.convert_ids_to_tokens(par))

def last_probs(prompt):
    enc = tok(prompt, return_tensors="pt").to(dev)
    with torch.inference_mode():
        lg = model(**enc).logits[0, -1, :].float()
    p = torch.softmax(lg, -1)
    top = int(p.argmax())
    pair = torch.softmax(lg[[a_tok, b_tok]], -1)[0].item()
    return pair, float(p[a_tok] + p[b_tok]), top, float(p[top])

print("\nofficial (next token after prompt) vs two-step (prompt + ' (')")
res = []
for r in rows:
    ans = [txt(x) for x in r[ak]][:5]
    q = r[qk]; story = r[sk] if sk else ""
    for i, a in enumerate(ans):
        others = ans[:i] + ans[i+1:]
        pr = se._build_prompt(q, a, others, story=story, max_brainstorm=10)
        po, mo, t1, p1 = last_probs(pr)
        pt, mt, t2, p2 = last_probs(pr + " (")
        res.append((po, mo, pt, mt))
        print(f"off P(true)={po:.3f} massAB={mo:.4f} top1={tok.convert_ids_to_tokens(t1)!r}({p1:.2f}) | "
              f"2step P(true)={pt:.3f} massAB={mt:.4f} top1={tok.convert_ids_to_tokens(t2)!r}({p2:.2f})")
import numpy as np
a = np.array(res)
print("\nMEAN  off P(true) %.3f sd %.3f massAB %.4f | 2step P(true) %.3f sd %.3f massAB %.4f"
      % (a[:,0].mean(), a[:,0].std(), a[:,1].mean(), a[:,2].mean(), a[:,2].std(), a[:,3].mean()))
