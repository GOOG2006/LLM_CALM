"""探针加固扫描: 一次前向取多层隐状态+NLL, 对每个配置跑案级5折LR-CV。
扫 层{28,24,20,16} / NLL-only / 最佳层+NLL融合。 一次加载模型跑多个数据集。
用法: python normprobe_sweep.py <model> <data_dir1> [data_dir2 ...]
"""
import json, glob, os, sys, math, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
DATAS = sys.argv[2:]
LAYERS = [-1, -5, -9, -13]          # 28,24,20,16 (hidden_states index; 0=emb)
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


def extract(full_ids, bounds):
    ids = torch.tensor([full_ids], device='cuda')
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        feats = {}
        for L in LAYERS:
            hs = out.hidden_states[L][0]
            feats[L] = torch.stack([hs[b - 1] for (a, b) in bounds]).float().cpu().numpy()
        lp = torch.log_softmax(out.logits[0][:-1].float(), dim=-1)
        tok_lp = lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
        nll = []
        for (a, b) in bounds:
            js = [j for j in range(max(a, 1), b)]
            nll.append(float(-tok_lp[[j - 1 for j in js]].mean()) if js else 0.0)
    return feats, np.array(nll, dtype=np.float32)[:, None]


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def train_lr(X, y, d, epochs=400, lr=0.05, l2=1e-2):
    Xt = torch.tensor(X, device='cuda'); yt = torch.tensor(y, device='cuda', dtype=torch.float32)
    w = torch.zeros(d, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr); lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(Xt @ w + b, yt) + l2 * (w * w).sum()
        loss.backward(); opt.step()
    return w.detach(), b.detach()


def cv_eval(cases):
    """cases: list of (label, X[nst,d]); case-level 5-fold. returns (auc, f1)."""
    d = cases[0][1].shape[1]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cases))
    fold = {int(idx): i % KFOLD for i, idx in enumerate(order)}
    scores = [None] * len(cases)
    for fk in range(KFOLD):
        Xtr, ytr = [], []
        for ci, (lab, F) in enumerate(cases):
            if fold[ci] == fk:
                continue
            for i in range(F.shape[0]):
                if lab == -1 or i < lab:
                    Xtr.append(F[i]); ytr.append(0)
                elif i == lab:
                    Xtr.append(F[i]); ytr.append(1)
        Xtr = np.stack(Xtr); ytr = np.array(ytr, np.float32)
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        w, b = train_lr((Xtr - mu) / sd, ytr, d)
        for ci, (lab, F) in enumerate(cases):
            if fold[ci] != fk:
                continue
            Xz = torch.tensor((F - mu) / sd, device='cuda', dtype=torch.float32)
            scores[ci] = torch.sigmoid(Xz @ w + b).cpu().numpy().tolist()
    pos, neg = [], []
    for ci, (lab, F) in enumerate(cases):
        p = scores[ci]
        for i in range(len(p)):
            if lab == -1 or i < lab:
                neg.append(p[i])
            elif i == lab:
                pos.append(p[i])
    best = 0.0
    for t100 in range(5, 100, 2):
        t = t100 / 100
        eh = et = ch = ct = 0
        for ci, (lab, F) in enumerate(cases):
            pred = next((i for i, pp in enumerate(scores[ci]) if pp > t), -1)
            if lab == -1:
                ct += 1; ch += (pred == -1)
            else:
                et += 1; eh += (pred == lab)
        ae = eh / max(et, 1); ac = ch / max(ct, 1)
        best = max(best, 2 * ae * ac / max(ae + ac, 1e-9))
    return auc(pos, neg), best * 100


for DATA in DATAS:
    folders = [f for f in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(f)]
    raw = []   # (label, feats_dict, nll)
    for f in folders:
        d = json.load(open(os.path.join(f, 'sample.json')))
        fids, bounds = build(d)
        if len(bounds) < 1:
            continue
        lab = d.get('label', -1)
        if lab is not None and lab >= len(bounds):
            lab = -1
        fe, nll = extract(fids, bounds)
        raw.append((lab, fe, nll))
    print(f"\n===== {DATA}  (model={MODEL}, cases={len(raw)}) =====")
    results = {}
    for L in LAYERS:
        a, f1 = cv_eval([(lab, fe[L]) for lab, fe, nll in raw])
        results[L] = a
        print(f"  layer {L:>3}         AUC={a:.3f}  F1={f1:.1f}")
    a, f1 = cv_eval([(lab, nll) for lab, fe, nll in raw])
    print(f"  NLL-only         AUC={a:.3f}  F1={f1:.1f}")
    bestL = max(results, key=results.get)
    a, f1 = cv_eval([(lab, np.concatenate([fe[bestL], nll], 1)) for lab, fe, nll in raw])
    print(f"  layer {bestL} + NLL  AUC={a:.3f}  F1={f1:.1f}   <- fusion(best layer)")
