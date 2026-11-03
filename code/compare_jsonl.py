import sys, json
def load(p, n=None):
    out = {}
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); out[str(r["id"])] = r
            if n and len(out) >= n: break
    return out
def walk(a, b, path, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b: walk(a[k], b[k], path + "/" + str(k), acc)
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, path + f"[{i}]", acc)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(a - b)
        if d > acc["max"]: acc["max"], acc["where"] = d, path
new, ref = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else None
A, B = load(new, n), load(ref)
common = [i for i in A if i in B]; acc = {"max": 0.0, "where": ""}
for i in common: walk(A[i], B[i], i, acc)
print(f"compared {len(common)} ids ({len(A)} in new); max abs diff {acc['max']:.3e} at {acc['where'] or '-'}")
sys.exit(1 if (acc["max"] > 1e-6 or not common) else 0)
