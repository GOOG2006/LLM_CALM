"""
Verify-by-Measurement 迭代3:隐藏态线性探针(测内部正确性表示,非表面流畅度)。
对每步 mean-pool 基座隐藏态(多层),用 numpy L2-逻辑回归 + 5折CV 测 error-AUROC。
仍属"测量"(读内部状态)+ 极轻线性读出(<LoRA)。报 CV AUROC(非训练),强正则防过拟合。
用法: python measure_hidden.py --model <path> --n_err 60 --n_cor 60
"""
import os, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PB_DIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SUBSETS = ["math"]
LAYERS = [8, 14, 20, 26, 28]


def load_cases(n_err, n_cor):
    err, cor = [], []
    for sub in SUBSETS:
        p = os.path.join(PB_DIR, f"{sub}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            lab = int(r["label"]); steps = r["steps"]
            if not isinstance(steps, list) or len(steps) < 2:
                continue
            (cor if lab == -1 else err).append((sub, r["id"], r["problem"], steps, lab))
    return err[:n_err], cor[:n_cor]


def auroc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列:平均秩
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    R = ranks[labels == 1].sum()
    return (R - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def train_lr(X, y, lam=1.0, iters=800, lr=0.5):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        gw = X.T @ g / n + lam * w / n
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b


def cv_auroc(X, y, k=5, lam=1.0):
    n = len(y)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    preds = np.zeros(n);
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        w, b = train_lr(Xtr, y[tr], lam=lam)
        preds[te] = Xte @ w + b
    return auroc(preds, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_err", type=int, default=60)
    ap.add_argument("--n_cor", type=int, default=60)
    ap.add_argument("--out", default="results_hidden.npz")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
                                                 output_hidden_states=True).cuda().eval()
    err, cor = load_cases(a.n_err, a.n_cor)
    cases = err + cor
    print(f"cases: err={len(err)} cor={len(cor)}", flush=True)

    feats = {L: [] for L in LAYERS}
    labels, cids, ts = [], [], []
    for (sub, cid, q, steps, lab) in cases:
        n = len(steps)
        for t in range(n):
            if lab >= 0 and t > lab:
                continue
            prev = "".join(f"Step {j+1}: {steps[j]}\n" for j in range(t)) or "(no previous steps)\n"
            prefix = f"Problem:\n{q}\n\nStep-by-step solution:\n{prev}Step {t+1}: "
            pre_ids = tok(prefix, add_special_tokens=True).input_ids
            stp_ids = tok(steps[t], add_special_tokens=False).input_ids
            if len(stp_ids) == 0:
                continue
            ids = pre_ids + stp_ids
            if len(ids) > 3500:
                keep = max(1, 3500 - len(stp_ids)); ids = pre_ids[-keep:] + stp_ids; pre_len = keep
            else:
                pre_len = len(pre_ids)
            inp = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                hs = model(inp).hidden_states  # tuple len n_layers+1, each [1,T,H]
            for L in LAYERS:
                v = hs[L][0, pre_len:len(ids)].float().mean(0).cpu().numpy()  # mean-pool step tokens
                feats[L].append(v)
            labels.append(int(lab >= 0 and t == lab)); cids.append(cid); ts.append(t)

    y = np.array(labels)
    print(f"steps={len(y)} pos={int(y.sum())}\n", flush=True)
    np.savez(a.out, y=y, cids=cids, ts=ts, **{f"L{L}": np.array(feats[L]) for L in LAYERS})

    print("===== 隐藏态线性探针 (5折CV error-AUROC) =====", flush=True)
    best = (0, None, None)
    for L in LAYERS:
        X = np.array(feats[L])
        for lam in [0.3, 1.0, 3.0]:
            au = cv_auroc(X, y, k=5, lam=lam)
            if au > best[0]:
                best = (au, L, lam)
            print(f"  layer {L:>2}  lam={lam}  AUROC={au:.3f}", flush=True)
    print(f"\n  BEST: layer {best[1]} lam={best[2]} AUROC={best[0]:.3f}", flush=True)
    print("  (对照: 表面熵 mean_ent≈0.68, 裸Yes/No探针≈0.46, GenPRM≈0.85)", flush=True)


if __name__ == "__main__":
    main()
