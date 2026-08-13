"""方法B v3: 冻结基座 + 层-9 + MLP头 + 特征缓存 + 域适配判决(A/B/C/BC)。
训练=GenPRM MATH-train(每步Yes/No), 固定阈值在训练集内选, 评ProcessBench四子集。
特征缓存到 feats_train_{n}.npz / feats_eval_{name}.npz —— 有缓存则不加载大模型。
用法: python train_probe_v3.py [n_train] [hidden]  (hidden=0 -> 线性)
"""
import json, glob, os, sys, re, torch, numpy as np
import torch.nn as nn

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 4096
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
HID = int(sys.argv[2]) if len(sys.argv) > 2 else 128
EVAL_DIRS = ['pb_gsm8k_120_in', 'pb_math_120_in', 'pb_olympiadbench_120_in', 'pb_omnimath_big_in']
_model = None


def get_model():
    global _model, tok
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    return _model


def build(problem, steps):
    steps = [s for s in steps if s is not None]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    ids = list(tok(problem + "\n", add_special_tokens=True).input_ids)
    bounds = []
    for s in steps:
        a = len(ids); ids += tok(s + "\n", add_special_tokens=False).input_ids
        bounds.append((a, len(ids)))
    if len(ids) > MAXLEN:
        ids = ids[:MAXLEN]; bounds = [(a, min(b, MAXLEN)) for a, b in bounds if a < MAXLEN]
    return ids, bounds


def feats(problem, steps):
    m = get_model()
    ids, bounds = build(problem, steps)
    if len(bounds) < 1:
        return None
    t = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        hs = m(t, output_hidden_states=True).hidden_states[LAYER][0]
        return torch.stack([hs[b - 1] for (a, b) in bounds]).float().cpu().numpy()


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']
    asts = [m['content'] for m in conv if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    labels = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0])
    problem, step0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    steps = [step0] + users[1:n]
    return problem, steps, labels[:len(steps)]


def get_train():
    cache = f'feats_train_{N_TRAIN}.npz'
    if os.path.exists(cache):
        d = np.load(cache); return d['X'], d['y']
    import pyarrow.parquet as pq
    tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
    idx = np.random.default_rng(0).permutation(len(tbl))[:N_TRAIN]
    Xs, ys = [], []
    for k, i in enumerate(idx):
        p, s, lab = parse_conv(tbl[int(i)]['conversations'])
        if len(s) < 1 or len(lab) != len(s):
            continue
        F = feats(p, s)
        if F is None or F.shape[0] != len(lab):
            continue
        Xs.append(F); ys.append(np.array(lab, np.float32))
        if (k + 1) % 1000 == 0:
            print(f"  train feats {k+1}/{len(idx)}", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    np.savez(cache, X=X, y=y)
    return X, y


def get_eval(DATA):
    cache = f"feats_eval_{os.path.basename(DATA)}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        F = d['F']; lens = d['lens']; labs = d['labs']
        cases, o = [], 0
        for L, lb in zip(lens, labs):
            cases.append((int(lb), F[o:o + L])); o += L
        return cases
    cases, Fall, lens, labs = [], [], [], []
    for f in [x for x in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(x)]:
        d = json.load(open(os.path.join(f, 'sample.json')))
        F = feats(d['problem'], d['steps'])
        if F is None:
            continue
        lab = d.get('label', -1)
        if lab is not None and lab >= F.shape[0]:
            lab = -1
        cases.append((lab, F)); Fall.append(F); lens.append(F.shape[0]); labs.append(lab)
    np.savez(cache, F=np.concatenate(Fall), lens=np.array(lens), labs=np.array(labs))
    return cases


# ---------- data ----------
X, y = get_train()
print(f"train steps={len(y)} err_rate={y.mean():.3f} n_train={N_TRAIN} hid={HID}", flush=True)
rng = np.random.default_rng(1); perm = rng.permutation(len(y)); cut = int(len(y) * 0.9)
tr, va = perm[:cut], perm[cut:]
mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
Xz = ((X - mu) / sd).astype(np.float32); H = X.shape[1]


class Head(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Linear(H, 1) if HID == 0 else nn.Sequential(nn.Linear(H, HID), nn.ReLU(), nn.Dropout(0.2), nn.Linear(HID, 1))

    def forward(s, x):
        return s.net(x).squeeze(-1)


dev = 'cuda'
net = Head().to(dev)
Xt = torch.tensor(Xz[tr], device=dev); yt = torch.tensor(y[tr], device=dev)
Xv = torch.tensor(Xz[va], device=dev); yv = y[va]
posw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), device=dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss(pos_weight=posw)


def auc(s, lab):
    p = s[lab == 1]; n = s[lab == 0]
    if len(p) == 0 or len(n) == 0:
        return 0.5
    return float((p[:, None] > n[None, :]).mean() * 1.0 + 0.5 * (p[:, None] == n[None, :]).mean())


best_auc, best_state = -1, None
for ep in range(400):
    net.train(); opt.zero_grad()
    loss = lossf(net(Xt), yt); loss.backward(); opt.step()
    if ep % 20 == 0 or ep == 399:
        net.eval()
        with torch.no_grad():
            a = auc(net(Xv).cpu().numpy(), yv)
        if a > best_auc:
            best_auc = a; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state); net.eval()
print(f"val AUC={best_auc:.3f}", flush=True)


def prob(Fz):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Fz, device=dev, dtype=torch.float32))).cpu().numpy()


pv = prob(Xz[va]); pm, psd = float(pv.mean()), float(pv.std() + 1e-9)


def pick(scores, lab):
    bt, bf = 0.5, -1
    for t in np.arange(-3, 3.01, 0.1):
        pred = (scores > t).astype(int)
        tp = ((pred == 1) & (lab == 1)).sum(); fp = ((pred == 1) & (lab == 0)).sum(); fn = ((pred == 0) & (lab == 1)).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f = 2 * pr * rc / max(pr + rc, 1e-9)
        if f > bf:
            bf, bt = f, t
    return bt


t_abs = pick(pv, yv)
zt = pick((pv - pm) / psd, yv)
print(f"t_abs={t_abs:.2f} zt={zt:.2f}", flush=True)


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


print("\n===== v3 (frozen DSR1 + L-9 + %s head, n_train=%d) =====" % ('linear' if HID == 0 else f'MLP{HID}', N_TRAIN))
print(f"{'dataset':16s} {'A:abs':>6} {'B:dom':>6} {'C:cz':>6} {'BC':>6}")
for D in EVAL_DIRS:
    cases = get_eval(D)
    allF = np.concatenate([F for _, F in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
    A = f1(cases, lambda F: next((i for i, p in enumerate(prob((F - mu) / sd)) if p > t_abs), -1))
    B = f1(cases, lambda F: next((i for i, p in enumerate(prob((F - mud) / sdd)) if p > t_abs), -1))
    C = f1(cases, lambda F: next((i for i, z in enumerate((prob((F - mu) / sd) - pm) / psd) if z > zt), -1))
    BC = f1(cases, lambda F: next((i for i, z in enumerate((prob((F - mud) / sdd) - pm) / psd) if z > zt), -1))
    print(f"{D:16s} {A:6.1f} {B:6.1f} {C:6.1f} {BC:6.1f}", flush=True)
