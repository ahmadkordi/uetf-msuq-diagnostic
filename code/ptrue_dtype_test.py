import sys, json, torch, numpy as np
import self_eval as se
dtype, out = sys.argv[2], sys.argv[3]
rows=[json.loads(l) for l in open(sys.argv[1])][:20]
tok, model, dev, a, b = se.load_eval_model("/mnt/data/models/Llama-2-13b-hf", dtype=dtype)
res=[]
for r in rows:
    ans=[x if isinstance(x,str) else next(v for v in x.values() if isinstance(v,str)) for x in r["responses"]][:5]
    for i,x in enumerate(ans):
        pr=se._build_prompt(r["question"], x, ans[:i]+ans[i+1:], story="", max_brainstorm=10)
        enc=tok(pr, return_tensors="pt").to(dev)
        with torch.inference_mode():
            lg=model(**enc).logits[0,-1,:].float()
        res.append(float(torch.softmax(lg[[a,b]],-1)[0]))
json.dump(res, open(out,"w"))
print(dtype, "mean %.3f sd %.3f distinct %d" % (np.mean(res), np.std(res), len(set(np.round(res,6)))))
