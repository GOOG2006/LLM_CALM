"""CORAL 协方差对齐(无监督域适配): 把证明域标准化特征的协方差对齐到 MATH 源域, 再套探针。
对比 B(仅mean/std) vs CORAL。复用缓存 feats_train_convs.pkl + feats_eval_*.npz(层-9)。
"""
import json, glob, os, pickle, torch, numpy as np

EVAL_DIRS = ['pb_gsm8k_120_in', 'pb_math_120_in', 'pb_olympiadbench_120_in', 'pb_omnimath_big_in']

convs = [c for c in pickle.load(open('feats_train_convs.pkl', 'rb')) if c is not None]
X = np.concatenate([F for _, F in convs]); y = np.concatenate([lab for lab, _ in convs])
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

# source standardized covariance^(1/2)  (in the standardized space, source ~ identity-ish but not exactly)
Zs = torch.tensor(Xz, device='cuda')
Cs = torch.cov(Zs.T) + 1e-3 * torch.eye(H, device='cuda')


def mat_pow(C, p):
    ev, V = torch.linalg.eigh(C)
    ev = torch.clamp(ev, min=1e-6)
    return (V * ev.pow(p)) @ V.T


Cs_half = mat_pow(Cs, 0.5)


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


def load_eval(DATA):
    d = np.load(f"feats_eval_{os.path.basename(DATA)}.npz")
    F, lens, labs = d['F'], d['lens'], d['labs']
    cases, o = [], 0
    for L, lb in zip(lens, labs):
        cases.append((int(lb), F[o:o + L])); o += L
    return cases


print(f"{'dataset':16s} {'B:mean/std':>10} {'CORAL':>8}")
for D in EVAL_DIRS:
    cases = load_eval(D)
    allF = np.concatenate([F for _, F in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
    # variant B
    B = f1(cases, lambda F: next((i for i, p in enumerate(prob((F - mud) / sdd)) if p > t_abs), -1))
    # CORAL: standardize target, whiten, recolor to source
    Zt_all = torch.tensor((allF - mud) / sdd, device='cuda', dtype=torch.float32)
    Ct = torch.cov(Zt_all.T) + 1e-3 * torch.eye(H, device='cuda')
    Wc = (mat_pow(Ct, -0.5) @ Cs_half).float()      # align target cov -> source cov

    def coral(F):
        Z = torch.tensor((F - mud) / sdd, device='cuda', dtype=torch.float32) @ Wc
        p = torch.sigmoid(Z @ w + b).cpu().numpy()
        return next((i for i, pp in enumerate(p) if pp > t_abs), -1)
    C = f1(cases, coral)
    print(f"{D:16s} {B:10.1f} {C:8.1f}", flush=True)
