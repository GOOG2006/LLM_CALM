"""隐状态案级交叉验证探针: 抽每步末token隐状态, 按案5折, 折内训LR判'该步是否错',
折外给测试案所有步打分 -> per-step AUC + 首错定位F1(扫阈值, 与NLL上界可比)。
用法: python normprobe_cv.py <data_dir> [limit] [model] [layer]
"""
import json, glob, os, sys, math, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA  = sys.argv[1] if len(sys.argv) > 1 else 'pb_math_120_in'
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'models/GenPRM-1.5B'
LAYER = int(sys.argv[4]) if len(sys.argv) > 4 else -1
MAXLEN, KFOLD = 4096, 5

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()


def build(sample):
    steps = [s for s in sample['steps'] if s is not None]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    full_ids = list(tok(sample['problem'] + "\n", add_special_tokens=True).input_ids)
    bounds = []
    for s in steps:
        start = len(full_ids)
        full_ids += tok(s + "\n", add_special_tokens=False).input_ids
        bounds.append((start, len(full_ids)))
    if len(full_ids) > MAXLEN:
        full_ids = full_ids[:MAXLEN]
        bounds = [(a, min(b, MAXLEN)) for a, b in bounds if a < MAXLEN]
    return full_ids, bounds


def feats_of(full_ids, bounds):
    ids = torch.tensor([full_ids], device='cuda')
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states[LAYER][0]  # [T,H]
        f = torch.stack([hs[b - 1] for (a, b) in bounds]).float().cpu().numpy()
    return f  # [nst, H]


cases = []   # (label, feats[nst,H])
folders = [f for f in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(f)][:LIMIT]
for f in folders:
    d = json.load(open(os.path.join(f, 'sample.json')))
    fids, bounds = build(d)
    if len(bounds) < 1:
        continue
    lab = d.get('label', -1)
    if lab is not None and lab >= len(bounds):
        lab = -1
    cases.append((lab, feats_of(fids, bounds)))

H = cases[0][1].shape[1]


def train_lr(X, y, epochs=400, lr=0.05, l2=1e-2):
    Xt = torch.tensor(X, device='cuda'); yt = torch.tensor(y, device='cuda', dtype=torch.float32)
    w = torch.zeros(H, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(Xt @ w + b, yt) + l2 * (w * w).sum()
        loss.backward(); opt.step()
    return w.detach(), b.detach()


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


# case-level 5-fold; probe never sees test cases
rng = np.random.default_rng(0)
order = rng.permutation(len(cases))
fold = {int(idx): i % KFOLD for i, idx in enumerate(order)}
scores_all = [None] * len(cases)          # per case: list of P(error) per step (out-of-fold)
for fk in range(KFOLD):
    Xtr, ytr = [], []
    for ci, (lab, F) in enumerate(cases):
        if fold[ci] == fk:
            continue
        nst = F.shape[0]
        for i in range(nst):
            if lab == -1:
                Xtr.append(F[i]); ytr.append(0)
            elif i < lab:
                Xtr.append(F[i]); ytr.append(0)
            elif i == lab:
                Xtr.append(F[i]); ytr.append(1)
            # i>lab: skip (ambiguous)
    Xtr = np.stack(Xtr); ytr = np.array(ytr, dtype=np.float32)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    w, b = train_lr((Xtr - mu) / sd, ytr)
    for ci, (lab, F) in enumerate(cases):
        if fold[ci] != fk:
            continue
        Xz = torch.tensor((F - mu) / sd, device='cuda', dtype=torch.float32)
        p = torch.sigmoid(Xz @ w + b).cpu().numpy()
        scores_all[ci] = p.tolist()

# --- per-step AUC (error vs correct, dropping ambiguous i>lab) ---
pos, neg = [], []
for ci, (lab, F) in enumerate(cases):
    p = scores_all[ci]
    for i in range(len(p)):
        if lab == -1 or i < lab:
            neg.append(p[i])
        elif i == lab:
            pos.append(p[i])
print(f"[{DATA}] model={MODEL} layer={LAYER}  cases={len(cases)}  err_steps={len(pos)} cor_steps={len(neg)}")
print(f"  per-step AUC (error detection) = {auc(pos, neg):.3f}   (0.5 = chance)")

# --- localization F1, sweep global threshold (comparable to NLL oracle) ---
best = (0.0, None)
for tau100 in range(5, 100, 2):
    tau = tau100 / 100
    eh = et = ch = ct = 0
    for ci, (lab, F) in enumerate(cases):
        p = scores_all[ci]
        pred = next((i for i, pp in enumerate(p) if pp > tau), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    f1 = 2 * ae * ac / max(ae + ac, 1e-9)
    if f1 > best[0]:
        best = (f1, (tau, ae, ac))
f1, (tau, ae, ac) = best
print(f"  localization F1 (CV, oracle-threshold) = {f1*100:.1f}  (tau={tau:.2f} accE={ae*100:.1f} accC={ac*100:.1f})")
