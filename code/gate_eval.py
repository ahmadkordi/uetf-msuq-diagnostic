#!/usr/bin/env python3
"""Stage J gate (v9 S12, S14) + baselines B1-B6 (v9 S18), evaluated with the OFFICIAL-protocol AUROC.

Implements, per cell (generator x dataset), on the 50/20/30 train/val/test id files:
  B1  best single estimator (chosen on TRAIN AUROC@90, frozen)
  B2  simple mean of e-CDF-calibrated estimators
  B3  original MS-UQ: SUM = U_Tucker+U_CP (paper eq.7) and MIN (eq.8 / shipped code), on RAW values
  B4  global weights: ONE softmax weight vector trained with the same KL-to-z_bar_SSL objective
      (our instantiation of v9's 'pre-specified differentiable pseudo-target'; flagged)
  B5  direct SSL heuristic: w_k = z_k^SSL / sum_j z_j^SSL                      (v9 S18.2)
  B5' SSL-only learned gate: input = [z_1^SSL..z_K^SSL]                         (v9 S18.3)
  B6  full gate: input = [z^SSL, phi_U (tilde U, mu, sigma, range, DC_i), phi_Q-lite]
Gate = FC(64)-ReLU-Dropout-FC(32)-FC(K) (literal v9 S12.1), loss KL(z_bar_SSL || softmax(r)) (S12.3, tau=1),
U_adaptive = sum_k w_k tilde U_k (S14). Training uses train ids only; early stopping on VAL KL (label-free).
(eta, alpha) are chosen on VAL AUROC@90 from a grid (v9 S15.3 lists them as validation-selected).

DOCUMENTED DEVIATIONS / OMISSIONS (v9 features not available or unspecified; never silently assumed):
  * phi_R (response-cluster features) omitted: v9 gives no clustering rule/threshold.
  * phi_S (source-matrix statistics) omitted in this version (matrices are not in the calibrated files).
  * phi_Q-lite = [log(1+words), TTR, 7 first-word task indicators]; NER count omitted. B6 is therefore
    'B6-partial' and must be labelled so in any report.
  * L_cons (S12.4) omitted: needs two bootstrap-derived views of the features. tau fixed to 1 (absorbed by r).
  * Selection of eta/alpha uses VAL labels (AUROC); the gate's TRAINING never sees labels.
  * B4's pseudo-target = z_bar_SSL (own choice).
TEST PROTOCOL: `--eval-split test` requires `--config <val-run json>` and `--confirm-test`; it never selects anything.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from baseline_auroc import official_auroc  # official-protocol AUROC (int() truncation, per-response expansion)

TASK_WORDS = ["who", "what", "where", "when", "why", "how"]
YN = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "has", "have", "had", "should"}


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln); out[str(r["id"])] = r
    return out


def read_ids(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def q_feats(question):
    w = re.findall(r"[A-Za-z0-9']+", question.lower())
    n = len(w); ttr = len(set(w)) / n if n else 0.0
    first = w[0] if w else ""
    task = [1.0 if first == t else 0.0 for t in TASK_WORDS] + [1.0 if first in YN else 0.0]
    return [np.log1p(n), ttr] + task


class Cell:
    def __init__(self, O: Path, cell: str, variant: str):
        D = O / "experiment_0b"; S = O / "experiment_0a" / "splits"
        cal = load(D / f"stage_g_trainfit_{cell}.jsonl"); ssl = load(D / f"ssl_{variant}_{cell}.jsonl")
        corr = load(O / f"stage_h_correctness_{cell}.jsonl")
        self.split = {s: [i for i in read_ids(S / f"{cell}.{s}_ids.txt") if i in cal and i in ssl and i in corr and corr[i].get("per_response")] for s in ("train", "val", "test")}
        ids = self.split["train"] + self.split["val"] + self.split["test"]
        names = sorted(set(cal[ids[0]]["calibrated"]) & set(ssl[ids[0]]["hat_a_k"]))
        self.names = names; self.K = len(names); self.index = {i: n for n, i in enumerate(ids)}
        self.U = np.array([[cal[i]["calibrated"][n] for n in names] for i in ids])
        self.HA = np.array([[ssl[i]["hat_a_k"][n] for n in names] for i in ids])
        self.HS = np.array([[ssl[i]["hat_s_k"][n] for n in names] for i in ids])
        self.Q = np.array([q_feats(ssl[i].get("question", "")) for i in ids])
        self.rawCP = np.array([cal[i]["raw"]["tensor.U_CP"] for i in ids]); self.rawTK = np.array([cal[i]["raw"]["tensor.U_Tucker"] for i in ids])
        self.N = len(corr[ids[0]]["per_response"])
        self.acc = {t: np.array([sum(1 for r in corr[i]["per_response"] if r["score"] > t) / len(corr[i]["per_response"]) for i in ids]) for t in (90, 70)}
        self.rows = {s: np.array([self.index[i] for i in self.split[s]]) for s in self.split}

    def auroc(self, u, rows, thr=90):
        return official_auroc(u, self.acc[thr][rows], self.N)

    def z(self, eta):
        return eta * self.HA + (1 - eta) * self.HS

    def zbar(self, eta, alpha):
        z = self.z(eta); return (z + alpha / self.K) / (z.sum(1, keepdims=True) + alpha)

    def feats_B6(self, eta):
        U = self.U; mu = U.mean(1, keepdims=True); sd = U.std(1, keepdims=True); rg = U.max(1, keepdims=True) - U.min(1, keepdims=True)
        dc = np.abs(U[:, :, None] - U[:, None, :]).sum(2) / (self.K - 1)
        return np.hstack([self.z(eta), U, mu, sd, rg, dc, self.Q])


# ---------- tiny numpy MLP (v9 S12.1 literal): FC64-ReLU-Dropout-FC32-FC(K); softmax; KL loss ----------
class Gate:
    def __init__(self, d, K, seed, hidden=(64, 32), drop=0.1):
        r = np.random.default_rng(seed); self.r = r; self.drop = drop; h1, h2 = hidden
        self.P = {"W1": r.normal(0, np.sqrt(2 / d), (d, h1)), "b1": np.zeros(h1), "W2": r.normal(0, np.sqrt(1 / h1), (h1, h2)), "b2": np.zeros(h2),
                  "W3": r.normal(0, np.sqrt(1 / h2), (h2, K)), "b3": np.zeros(K)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}; self.v = {k: np.zeros_like(v) for k, v in self.P.items()}; self.t = 0

    def fwd(self, X, train=False):
        P = self.P; a1 = X @ P["W1"] + P["b1"]; h1 = np.maximum(a1, 0)
        mask = None
        if train and self.drop > 0:
            mask = (self.r.random(h1.shape) > self.drop) / (1 - self.drop); h1 = h1 * mask
        h2 = h1 @ P["W2"] + P["b2"]; r = h2 @ P["W3"] + P["b3"]
        e = np.exp(r - r.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        return p, (X, a1, h1, h2, mask)

    def step(self, X, q, lr=1e-3, wd=1e-4):
        p, (X, a1, h1, h2, mask) = self.fwd(X, True); n = len(X); dr = (p - q) / n; P = self.P
        g = {"W3": h2.T @ dr, "b3": dr.sum(0)}; dh2 = dr @ P["W3"].T
        g["W2"] = h1.T @ dh2; g["b2"] = dh2.sum(0); dh1 = dh2 @ P["W2"].T
        if mask is not None: dh1 = dh1 * mask
        da1 = dh1 * (a1 > 0); g["W1"] = X.T @ da1; g["b1"] = da1.sum(0)
        self.t += 1
        for k in P:
            gk = g[k] + (wd * P[k] if k.startswith("W") else 0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk; self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            P[k] -= lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)


def kl(q, p):
    return float(np.mean(np.sum(q * (np.log(q + 1e-12) - np.log(p + 1e-12)), 1)))


def train_gate(Xtr, qtr, Xva, qva, K, seed, max_ep=120, patience=10, bs=128):
    g = Gate(Xtr.shape[1], K, seed); rng = np.random.default_rng(seed + 1000); best = (1e9, None, 0); bad = 0
    for ep in range(max_ep):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]; g.step(Xtr[b], qtr[b])
        v = kl(qva, g.fwd(Xva)[0])
        if v < best[0] - 1e-6: best = (v, {k: x.copy() for k, x in g.P.items()}, ep); bad = 0
        else:
            bad += 1
            if bad >= patience: break
    g.P = best[1]; return g, best[0], best[2]


def global_weights(qtr, qva, K, max_ep=400):
    gv = np.zeros(K); m = np.zeros(K); v = np.zeros(K); best = (1e9, gv.copy())
    for t in range(1, max_ep + 1):
        p = np.exp(gv - gv.max()); p /= p.sum(); grad = p - qtr.mean(0)
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2; gv -= 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        p = np.exp(gv - gv.max()); p /= p.sum(); kv = kl(qva, np.tile(p, (len(qva), 1)))
        if kv < best[0]: best = (kv, gv.copy())
    p = np.exp(best[1] - best[1].max()); return p / p.sum()


def entropy_norm(w):
    return float(np.mean(-(w * np.log(w + 1e-12)).sum(1) / np.log(w.shape[1])))


def predict_models(C: Cell, eta, alpha, rows_eval, seeds, which):
    """train on TRAIN, early-stop on VAL KL (label-free); return {model: (U_eval[Q], weights/None, info)}"""
    tr, va = C.rows["train"], C.rows["val"]; qbar = C.zbar(eta, alpha); out = {}
    if "B5" in which:
        z = C.z(eta); w = z / z.sum(1, keepdims=True); out["B5"] = ((w * C.U)[rows_eval].sum(1), w[rows_eval], {})
    if "B4" in which:
        w = global_weights(qbar[tr], qbar[va], C.K); out["B4"] = ((C.U[rows_eval] * w).sum(1), np.tile(w, (len(rows_eval), 1)), {})
    for name, X in (("B5'", C.z(eta)), ("B6", C.feats_B6(eta))):
        if name not in which: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - mu) / sd; Us, Ws, eps = [], [], []
        for s in seeds:
            g, kv, ep = train_gate(Xs[tr], qbar[tr], Xs[va], qbar[va], C.K, s)
            w = g.fwd(Xs[rows_eval])[0]; Ws.append(w); Us.append((w * C.U[rows_eval]).sum(1)); eps.append((kv, ep))
        out[name] = (np.mean(Us, 0), np.mean(Ws, 0), {"seed_U": Us, "val_kl": [e[0] for e in eps], "epochs": [e[1] for e in eps]})
    return out


def paired_boot(C, rows, thr, uA, uB, n_boot, seed):
    rng = np.random.default_rng(seed); n = len(rows); d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); d.append(official_auroc(uA[idx], C.acc[thr][rows][idx], C.N) - official_auroc(uB[idx], C.acc[thr][rows][idx], C.N))
    d = np.array(d); return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/mnt/output")); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--variant", default="cal", choices=["cal", "raw"]); ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--config", type=Path, default=None); ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]); ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--n-boot", type=int, default=500); ap.add_argument("--tag", default="gate")
    a = ap.parse_args()
    if a.eval_split == "test" and not (a.config and a.confirm_test): sys.exit("TEST split needs --config (from a val run) and --confirm-test. Nothing is selected on test.")
    O = a.data_dir; outdir = O / "experiment_0b" / "gate"; outdir.mkdir(parents=True, exist_ok=True); seeds = list(range(a.seeds))
    cfg_all = json.loads(a.config.read_text()) if a.config else {}; results = {}; lines = []
    for cell in a.cells:
        C = Cell(O, cell, a.variant); tr, va, te = C.rows["train"], C.rows["val"], C.rows["test"]; ev = va if a.eval_split == "val" else te
        lines.append(f"\n== {cell}  split={a.eval_split} nTrain={len(tr)} nVal={len(va)} nTest={len(te)} K={C.K} =="); cfg = cfg_all.get(cell, {}) if a.config else {}
        if a.eval_split == "val":
            sel = {}
            for m in ("B4", "B5", "B5'", "B6"):
                best = (-1, None)
                for eta in a.etas:
                    for al in (a.alphas if m in ("B4", "B5'", "B6") else [1.0]):
                        o = predict_models(C, eta, al, va, seeds, {m})[m][0]; sc = C.auroc(o, va, 90)
                        if sc > best[0]: best = (sc, (eta, al))
                sel[m] = {"eta": best[1][0], "alpha": best[1][1], "val_auroc90": best[0]}
            b1 = max(range(C.K), key=lambda k: C.auroc(C.U[tr][:, k], tr, 90)); sel["B1"] = {"name": C.names[b1]}
            b1_70 = max(range(C.K), key=lambda k: C.auroc(C.U[tr][:, k], tr, 70)); sel["B1@70"] = {"name": C.names[b1_70]}
            cfg = sel
        results[cell] = cfg
        U_by = {"B1": C.U[ev][:, C.names.index(cfg["B1"]["name"])], "B1@70": C.U[ev][:, C.names.index(cfg["B1@70"]["name"])], "B2": C.U[ev].mean(1), "B3-Sum": (C.rawTK + C.rawCP)[ev], "B3-Min": np.minimum(C.rawTK, C.rawCP)[ev]}
        info = {}
        for m in ("B4", "B5", "B5'", "B6"):
            o = predict_models(C, cfg[m]["eta"], cfg[m]["alpha"], ev, seeds, {m})[m]; U_by[m] = o[0]; info[m] = o
        lines.append(f"  {'method':8s} {'AUROC@90':>9s} {'AUROC@70':>9s}  notes")
        for m, u in U_by.items():
            extra = ""
            if m in cfg and "eta" in cfg[m]: extra = f"eta={cfg[m]['eta']:g} alpha={cfg[m]['alpha']:g}"
            if m in ("B1", "B1@70"): extra = cfg[m]["name"]
            if m in info and info[m][1] is not None: extra += f" H_w={entropy_norm(info[m][1]):.3f}"
            if m in info and info[m][2].get("seed_U"):
                sd = [C.auroc(s_, ev, 90) for s_ in info[m][2]["seed_U"]]; extra += f" seeds90 {np.mean(sd):.4f}±{np.std(sd):.4f}"
            lines.append(f"  {m:8s} {C.auroc(u, ev, 90):9.4f} {C.auroc(u, ev, 70):9.4f}  {extra}")
        for A, B in (("B6", "B5'"), ("B5'", "B5"), ("B5", "B2"), ("B6", "B1"), ("B6", "B1@70"), ("B6", "B2"), ("B6", "B3-Sum")):
            for thr in (90, 70):
                d = C.auroc(U_by[A], ev, thr) - C.auroc(U_by[B], ev, thr); lo, hi = paired_boot(C, ev, thr, U_by[A], U_by[B], a.n_boot, 1)
                lines.append(f"  paired {A} - {B} @ {thr}: {d:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
        print("\n".join(lines[-(len(U_by) + 16):]), flush=True)
    (outdir / f"{a.tag}_{a.eval_split}_{'_'.join(a.cells) if len(a.cells) < 3 else 'multi'}.txt").write_text("\n".join(lines))
    if a.eval_split == "val": (outdir / f"{a.tag}_config_{'_'.join(a.cells) if len(a.cells) < 3 else 'multi'}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
