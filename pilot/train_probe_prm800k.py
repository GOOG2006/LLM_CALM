"""方法B — 独立标签版: 冻结基座层-9 + 线性探针, 训练标签来自 PRM800K(人工, 无GenPRM)。
样本 prm800k_ex.jsonl: {problem, steps:[ctx...,cand], y=最后一步标签}。取最后一步特征。
评 ProcessBench 复用 feats_eval_pb_*.npz(层-9基座特征)。缓存训练特征 feats_prm800k.npz。
用法: python train_probe_prm800k.py [n]
"""
import json, glob, os, sys, torch, numpy as np

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 4096
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
EVAL_DIRS = ['pb_gsm8k_120_in', 'pb_math_120_in', 'pb_olympiadbench_120_in', 'pb_omnimath_big_in']
CACHE = 'feats_prm800k.npz'
_m = None


def get_model():
    global _m, tok
    if _m is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        _m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    return _m


def feat_last(problem, steps):
    """返回最后一步末token的层-9特征(单向量)。"""
    m = get_model()
    steps = [(s if isinstance(s, str) else (s.get('text', '') if isinstance(s, dict) else str(s))) for s in steps]
    steps = [s for s in steps if s]
    if not steps:
        return None
    ids = list(tok(problem + "\n", add_special_tokens=True).input_ids)
    last_start = None
    for i, s in enumerate(steps):
        if i == len(steps) - 1:
            last_start = len(ids)
        ids += tok(s + "\n", add_special_tokens=False).input_ids
    if len(ids) > MAXLEN:
        ids = ids[:MAXLEN]
    if last_start is None or last_start >= len(ids):
        return None
    t = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        hs = m(t, output_hidden_states=True).hidden_states[LAYER][0]
        return hs[len(ids) - 1].float().cpu().numpy()


def get_train():
    if os.path.exists(CACHE):
        d = np.load(CACHE); return d['X'], d['y']
    ex = [json.loads(l) for l in open('prm800k_ex.jsonl', encoding='utf-8')][:N]
    Xs, ys = [], []
    for k, e in enumerate(ex):
        f = feat_last(e['problem'], e['steps'])
        if f is None:
            continue
        Xs.append(f); ys.append(e['y'])
        if (k + 1) % 2000 == 0:
            print(f"  feats {k+1}/{len(ex)}", flush=True)
    X = np.stack(Xs); y = np.array(ys, np.float32)
    np.savez(CACHE, X=X, y=y)
    return X, y


def load_eval(DATA):
    d = np.load(f"feats_eval_{os.path.basename(DATA)}.npz")
    F, lens, labs = d['F'], d['lens'], d['labs']
    cases, o = [], 0
    for L, lb in zip(lens, labs):
        cases.append((int(lb), F[o:o + L])); o += L
    return cases


X, y = get_train()
print(f"train examples={len(y)} err_rate={y.mean():.3f}", flush=True)
rng = np.random.default_rng(1); perm = rng.permutation(len(y)); cut = int(len(y) * 0.9)
tr, va = perm[:cut], perm[cut:]
mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
Xz = ((X - mu) / sd).astype(np.float32); H = X.shape[1]
Xt = torch.tensor(Xz[tr], device='cuda'); yt = torch.tensor(y[tr], device='cuda')
w = torch.zeros(H, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.05)
lf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), device='cuda'))
for _ in range(600):
    opt.zero_grad(); loss = lf(Xt @ w + b, yt) + 1e-2 * (w * w).sum(); loss.backward(); opt.step()
w = w.detach(); b = b.detach()


def prob(Fz):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(Fz, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()


pv = prob(Xz[va]); yv = y[va]; pm, psd = float(pv.mean()), float(pv.std() + 1e-9)
_p = pv[yv == 1]; _n = pv[yv == 0]
vauc = float((_p[:, None] > _n[None, :]).mean()) if len(_p) and len(_n) else 0.5


def pick(sc):
    bt, bf = 0.5, -1
    for t in np.arange(-3, 3.01, 0.1):
        pr = (sc > t).astype(int)
        tp = ((pr == 1) & (yv == 1)).sum(); fp = ((pr == 1) & (yv == 0)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
        if f > bf:
            bf, bt = f, t
    return bt


t_abs = pick(pv); zt = pick((pv - pm) / psd)
print(f"val-AUC={vauc:.3f} t_abs={t_abs:.2f} zt={zt:.2f}", flush=True)


def f1(cases, fn):
    eh = et = ch = ct = 0
    for lab, F in cases:
        pred = fn(F)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


print(f"\n===== Method B — PRM800K labels (no GenPRM) =====")
print(f"{'dataset':16s} {'B:dom':>7} {'BC':>7}")
for D in EVAL_DIRS:
    cases = load_eval(D)
    allF = np.concatenate([F for _, F in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
    B = f1(cases, lambda F: next((i for i, p in enumerate(prob((F - mud) / sdd)) if p > t_abs), -1))
    BC = f1(cases, lambda F: next((i for i, z in enumerate((prob((F - mud) / sdd) - pm) / psd) if z > zt), -1))
    print(f"{D:16s} {B:7.1f} {BC:7.1f}", flush=True)
