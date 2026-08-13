"""阈值重标定: 在 MATH-train 整案上按'首错定位F1'选阈值(而非步级F1), 再在全量ProcessBench评测。
对比 步级F1阈值 vs 定位F1阈值。末token探针(诚实方法)。复用 feats_train_convs.pkl + feats_full_*.npz。
"""
import os, pickle, torch, numpy as np
FULL = ['pb_gsm8k_full_in', 'pb_math_full_in', 'pb_olympiadbench_full_in', 'pb_omnimath_full_in']
GEN = {'pb_gsm8k_full_in': 52.8, 'pb_math_full_in': 66.6, 'pb_olympiadbench_full_in': 55.1, 'pb_omnimath_full_in': 54.5}

convs = [c for c in pickle.load(open('feats_train_convs.pkl', 'rb')) if c is not None]
rng = np.random.default_rng(1)
order = rng.permutation(len(convs))
cut = int(len(convs) * 0.9)
tr_c = [convs[i] for i in order[:cut]]; va_c = [convs[i] for i in order[cut:]]
Xtr = np.concatenate([F for _, F in tr_c]); ytr = np.concatenate([lab for lab, _ in tr_c])
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
H = Xtr.shape[1]
Xt = torch.tensor(((Xtr - mu) / sd).astype(np.float32), device='cuda'); yt = torch.tensor(ytr, device='cuda')
w = torch.zeros(H, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.05)
lf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-3), device='cuda'))
for _ in range(600):
    opt.zero_grad(); loss = lf(Xt @ w + b, yt) + 1e-2 * (w * w).sum(); loss.backward(); opt.step()
w = w.detach(); b = b.detach()


def prob(Fz):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(Fz, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()


# ---- threshold 1: step-level F1 on val steps ----
Xv = np.concatenate([F for _, F in va_c]); yv = np.concatenate([lab for lab, _ in va_c])
pv = prob((Xv - mu) / sd)


def pick_step(sc, lab):
    bt, bf = 0.5, -1
    for t in np.arange(-3, 3.01, 0.05):
        pr = (sc > t).astype(int)
        tp = ((pr == 1) & (lab == 1)).sum(); fp = ((pr == 1) & (lab == 0)).sum(); fn = ((pr == 0) & (lab == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
        if f > bf:
            bf, bt = f, t
    return bt


t_step = pick_step(pv, yv)

# ---- threshold 2: localization F1 on val CONVERSATIONS ----
val_cases = []
for lab, F in va_c:
    fe = next((i for i, v in enumerate(lab) if v == 1), -1)   # gold first-error step (else -1)
    val_cases.append((fe, prob((F - mu) / sd)))


def loc_f1_at(t, cases):
    eh = et = ch = ct = 0
    for fe, P in cases:
        pred = next((i for i, p in enumerate(P) if p > t), -1)
        if fe == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == fe)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9)


t_loc, bf = 0.5, -1
for t in np.arange(0.02, 0.99, 0.02):
    f = loc_f1_at(t, val_cases)
    if f > bf:
        bf, t_loc = f, t
print(f"t_step(step-F1)={t_step:.2f}  t_loc(localization-F1)={t_loc:.2f}  val-loc-F1@t_loc={bf*100:.1f}", flush=True)


def load_full(DATA):
    d = np.load(f"feats_full_{os.path.basename(DATA)}.npz")
    Flt, lens, labs = d['Flt'], d['lens'], d['labs']
    cases, o = [], 0
    for L, lb in zip(lens, labs):
        cases.append((int(lb), Flt[o:o + L])); o += L
    return cases


def eval_full(cases, t):
    Fall = np.concatenate([F for _, F in cases]); md, sdd = Fall.mean(0), Fall.std(0) + 1e-6
    eh = et = ch = ct = 0
    for lab, F in cases:
        P = prob((F - md) / sdd)
        pred = next((i for i, p in enumerate(P) if p > t), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100, ae * 100, ac * 100


print(f"\n{'dataset':16s} {'@t_step':>8} {'@t_loc':>8} {'accE/accC@loc':>14}  GenPRM")
for D in FULL:
    cases = load_full(D)
    fs, _, _ = eval_full(cases, t_step)
    fl, ae, ac = eval_full(cases, t_loc)
    print(f"{os.path.basename(D):16s} {fs:8.1f} {fl:8.1f}   {ae:4.0f}/{ac:<4.0f}      {GEN[D]}", flush=True)
