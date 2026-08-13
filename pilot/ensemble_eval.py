"""集成 末token探针 + 均值池化探针。两者互补(math靠末token, 证明靠mean-pool)。
对比: 末token / 均值池化 / 平均集成 / 拼接。复用四份缓存(train_convs, train_mp, eval_*, eval_mp_*)。
"""
import glob, os, pickle, torch, numpy as np
EVAL_DIRS = ['pb_gsm8k_120_in', 'pb_math_120_in', 'pb_olympiadbench_120_in', 'pb_omnimath_big_in']


def train_probe(X, y):
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
    return w.detach(), b.detach(), mu, sd, va


def prob(Fz, w, b):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(Fz, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()


def load_train(cache):
    convs = [c for c in pickle.load(open(cache, 'rb')) if c is not None]
    return np.concatenate([F for _, F in convs]), np.concatenate([lab for lab, _ in convs])


def load_eval(cache):
    d = np.load(cache); F, lens, labs = d['F'], d['lens'], d['labs']
    cases, o = [], 0
    for L, lb in zip(lens, labs):
        cases.append((int(lb), F[o:o + L])); o += L
    return cases


Xlt, ylt = load_train('feats_train_convs.pkl')
Xmp, ymp = load_train('feats_train_mp.pkl')
n = min(len(ylt), len(ymp)); Xlt, ylt, Xmp, ymp = Xlt[:n], ylt[:n], Xmp[:n], ymp[:n]
wlt, blt, mult, sdlt, va = train_probe(Xlt, ylt)
wmp, bmp, mump, sdmp, _ = train_probe(Xmp, ymp)
Xcat = np.concatenate([((Xlt - mult) / sdlt), ((Xmp - mump) / sdmp)], 1).astype(np.float32)
wc, bc, muc, sdc, _ = train_probe(Xcat, ylt)   # already standardized -> muc~0

# thresholds on val
yv = ylt[va]
plt_v = prob(((Xlt - mult) / sdlt)[va], wlt, blt)
pmp_v = prob(((Xmp - mump) / sdmp)[va], wmp, bmp)
pav_v = (plt_v + pmp_v) / 2
pc_v = prob(Xcat[va], wc, bc)


def pick(sc):
    bt, bf = 0.5, -1
    for t in np.arange(-3, 3.01, 0.1):
        pr = (sc > t).astype(int)
        tp = ((pr == 1) & (yv == 1)).sum(); fp = ((pr == 1) & (yv == 0)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
        if f > bf:
            bf, bt = f, t
    return bt


t_lt, t_mp, t_av, t_c = pick(plt_v), pick(pmp_v), pick(pav_v), pick(pc_v)


def f1(cases, predfn):
    eh = et = ch = ct = 0
    for lab, F in cases:
        pred = predfn(F)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


print(f"{'dataset':16s} {'lastTok':>8} {'meanPool':>9} {'ensAvg':>7} {'confPick':>9}")
for D in EVAL_DIRS:
    clt = load_eval(f"feats_eval_{D}.npz")
    cmp = load_eval(f"feats_eval_mp_{D}.npz")
    Flt = np.concatenate([F for _, F in clt]); mdlt, sdd_lt = Flt.mean(0), Flt.std(0) + 1e-6
    Fmp = np.concatenate([F for _, F in cmp]); mdmp, sdd_mp = Fmp.mean(0), Fmp.std(0) + 1e-6

    def PLT(F): return prob((F - mdlt) / sdd_lt, wlt, blt)
    def PMP(F): return prob((F - mdmp) / sdd_mp, wmp, bmp)
    lt = f1(clt, lambda F: next((i for i, p in enumerate(PLT(F)) if p > t_lt), -1))
    mp = f1(cmp, lambda F: next((i for i, p in enumerate(PMP(F)) if p > t_mp), -1))
    # ensemble: align by case index (same order/steps)
    en = 0
    eh = et = ch = ct = 0
    for (lab, Fl), (_, Fm) in zip(clt, cmp):
        m = min(len(Fl), len(Fm))
        pav = (PLT(Fl[:m]) + PMP(Fm[:m])) / 2
        pred = next((i for i, p in enumerate(pav) if p > t_av), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); en = 2 * ae * ac / max(ae + ac, 1e-9) * 100
    # confidence-pick: per step use the probe more confident (further from 0.5)
    eh = et = ch = ct = 0
    for (lab, Fl), (_, Fm) in zip(clt, cmp):
        m = min(len(Fl), len(Fm))
        pl = PLT(Fl[:m]); pm2 = PMP(Fm[:m])
        err = []
        for i in range(m):
            if abs(pl[i] - 0.5) >= abs(pm2[i] - 0.5):
                err.append(pl[i] > t_lt)
            else:
                err.append(pm2[i] > t_mp)
        pred = next((i for i, e in enumerate(err) if e), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); cp = 2 * ae * ac / max(ae + ac, 1e-9) * 100
    # per-domain auto-select: pick feature more 'decisive' (higher mean |P-0.5|) on this domain (unsupervised)
    dec_lt = float(np.mean([abs(p - 0.5) for _, F in clt for p in PLT(F)]))
    dec_mp = float(np.mean([abs(p - 0.5) for _, F in cmp for p in PMP(F)]))
    auto = lt if dec_lt >= dec_mp else mp
    pick_name = 'lastTok' if dec_lt >= dec_mp else 'meanPool'
    print(f"{D:16s} {lt:8.1f} {mp:9.1f} {cp:7.1f}   auto={auto:5.1f}({pick_name} lt={dec_lt:.3f} mp={dec_mp:.3f})", flush=True)
