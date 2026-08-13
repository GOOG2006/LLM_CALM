"""上界闸门: 用一个证明子集当'证明域训练数据'替身, 帮另一个证明子集(留一法)。
测"有证明域步级标签"能把 olympiad/omnimath 拉高多少 -> 决定要不要投入搞真数据。
末token特征。MATH-train(feats_train_convs.pkl) + 证明代理子集(feats_full_*.npz)。
"""
import os, pickle, torch, numpy as np
GEN = {'gsm8k': 52.8, 'math': 66.6, 'olympiadbench': 55.1, 'omnimath': 54.5}


def load_math():
    convs = [c for c in pickle.load(open('feats_train_convs.pkl', 'rb')) if c is not None]
    X = np.concatenate([F for _, F in convs]); y = np.concatenate([lab for lab, _ in convs])
    return X, y


def load_full(cfg):
    d = np.load(f"feats_full_pb_{cfg}_full_in.npz")
    Flt, lens, labs = d['Flt'], d['lens'], d['labs']
    cases, o = [], 0
    for L, lb in zip(lens, labs):
        cases.append((int(lb), Flt[o:o + L])); o += L
    return cases


def cases_to_steps(cases):
    """first-error case-label -> per-step examples (i<fe:0, i==fe:1, i>fe:drop)."""
    Xs, ys = [], []
    for fe, F in cases:
        for i in range(len(F)):
            if fe == -1 or i < fe:
                Xs.append(F[i]); ys.append(0)
            elif i == fe:
                Xs.append(F[i]); ys.append(1)
    return np.stack(Xs), np.array(ys, np.float32)


def train(X, y):
    mu = X.mean(0); sd = X.std(0) + 1e-6; H = X.shape[1]
    Xt = torch.tensor(((X - mu) / sd).astype(np.float32), device='cuda'); yt = torch.tensor(y, device='cuda')
    w = torch.zeros(H, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    lf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y.mean()) / max(y.mean(), 1e-3), device='cuda'))
    for _ in range(600):
        opt.zero_grad(); loss = lf(Xt @ w + b, yt) + 1e-2 * (w * w).sum(); loss.backward(); opt.step()
    return w.detach(), b.detach(), mu, sd


def prob(Fz, w, b):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(Fz, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()


def eval_full(cases, w, b, t=0.5):
    Fall = np.concatenate([F for _, F in cases]); md, sdd = Fall.mean(0), Fall.std(0) + 1e-6
    eh = et = ch = ct = 0
    for lab, F in cases:
        P = prob((F - md) / sdd, w, b)
        pred = next((i for i, p in enumerate(P) if p > t), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100, ae * 100, ac * 100


Xm, ym = load_math()
oly = load_full('olympiadbench'); omn = load_full('omnimath')
Xoly, yoly = cases_to_steps(oly); Xomn, yomn = cases_to_steps(omn)

# MATH-only baseline
wo, bo, _, _ = train(Xm, ym)
base_oly = eval_full(oly, wo, bo)[0]; base_omn = eval_full(omn, wo, bo)[0]

# MATH + omnimath(proxy) -> eval olympiad
w1, b1, _, _ = train(np.concatenate([Xm, Xomn]), np.concatenate([ym, yomn]))
up_oly, ae1, ac1 = eval_full(oly, w1, b1)

# MATH + olympiad(proxy) -> eval omnimath
w2, b2, _, _ = train(np.concatenate([Xm, Xoly]), np.concatenate([ym, yoly]))
up_omn, ae2, ac2 = eval_full(omn, w2, b2)

print(f"olympiad:  MATH-only={base_oly:.1f}  +proof(omni)={up_oly:.1f}  (accE {ae1:.0f}/accC {ac1:.0f})  GenPRM={GEN['olympiadbench']}")
print(f"omnimath:  MATH-only={base_omn:.1f}  +proof(oly)={up_omn:.1f}  (accE {ae2:.0f}/accC {ac2:.0f})  GenPRM={GEN['omnimath']}")
