"""多层拼接探针: 层[-5,-9,-13]在每步末token拼接(4608维) -> 线性探针 + B/BC。
同一次运行对比: 拼接 vs 仅层-9切片。 用法: python train_probe_ml.py [n]
"""
import json, glob, os, sys, re, pickle, torch, numpy as np

MODEL = 'models/DSR1-1.5B'
LAYERS = [-5, -9, -13]
MID = 1                     # index of -9 within LAYERS (for single-layer slice)
MAXLEN = 4096
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
EVAL_DIRS = ['pb_gsm8k_120_in', 'pb_math_120_in', 'pb_olympiadbench_120_in', 'pb_omnimath_big_in']
CACHE = 'feats_train_ml.pkl'
_m = None


def get_model():
    global _m, tok
    if _m is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        _m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    return _m


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
        hsall = m(t, output_hidden_states=True).hidden_states
        per = []
        for (a, b) in bounds:
            per.append(torch.cat([hsall[L][0][b - 1] for L in LAYERS]).float().cpu().numpy())
    return np.stack(per)                      # [nst, len(LAYERS)*H]


def parse_conv(conv):
    users = [x['content'] for x in conv if x['role'] == 'user']
    asts = [x['content'] for x in conv if x['role'] == 'assistant']
    n = min(len(users), len(asts))
    labels = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0])
    problem, step0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    steps = [step0] + users[1:n]
    return problem, steps, labels[:len(steps)]


def get_convs(N):
    cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else []
    if len(cache) < N:
        import pyarrow.parquet as pq
        tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
        idx = np.random.default_rng(0).permutation(len(tbl))
        for pos in range(len(cache), N):
            p, s, lab = parse_conv(tbl[int(idx[pos])]['conversations'])
            F = feats(p, s) if (len(s) >= 1 and len(lab) == len(s)) else None
            cache.append((np.array(lab, np.float32), F) if (F is not None and F.shape[0] == len(lab)) else None)
            if (pos + 1) % 1000 == 0:
                print(f"  feats {pos+1}/{N}", flush=True); pickle.dump(cache, open(CACHE, 'wb'))
        pickle.dump(cache, open(CACHE, 'wb'))
    return [c for c in cache[:N] if c is not None]


def get_eval(DATA):
    cache = f"feats_evalml_{os.path.basename(DATA)}.npz"
    if os.path.exists(cache):
        d = np.load(cache); F, lens, labs = d['F'], d['lens'], d['labs']
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


convs = get_convs(N)
X = np.concatenate([F for _, F in convs]); y = np.concatenate([lab for lab, _ in convs])
Dtot = X.shape[1]; Hsub = Dtot // len(LAYERS)
print(f"train steps={len(y)} err={y.mean():.3f} dim={Dtot}", flush=True)


def evaluate(Xtr_cols):
    Xc = X[:, Xtr_cols]
    rng = np.random.default_rng(1); perm = rng.permutation(len(y)); cut = int(len(y) * 0.9)
    tr, va = perm[:cut], perm[cut:]
    mu = Xc[tr].mean(0); sd = Xc[tr].std(0) + 1e-6
    Xz = ((Xc - mu) / sd).astype(np.float32); H = Xc.shape[1]; dev = 'cuda'
    Xt = torch.tensor(Xz[tr], device=dev); yt = torch.tensor(y[tr], device=dev)
    w = torch.zeros(H, device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    posw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), device=dev)
    lf = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
    for _ in range(600):
        opt.zero_grad(); loss = lf(Xt @ w + b, yt) + 1e-2 * (w * w).sum(); loss.backward(); opt.step()
    w = w.detach(); b = b.detach()

    def prob(Fz):
        with torch.no_grad():
            return torch.sigmoid(torch.tensor(Fz, device=dev, dtype=torch.float32) @ w + b).cpu().numpy()
    pv = prob(Xz[va]); yv = y[va]; pm, psd = float(pv.mean()), float(pv.std() + 1e-9)
    _p = pv[yv == 1]; _n = pv[yv == 0]
    vauc = float((_p[:, None] > _n[None, :]).mean()) if len(_p) and len(_n) else 0.5

    def pick(sc):
        bt, bf = 0.5, -1
        for t in np.arange(-3, 3.01, 0.1):
            pred = (sc > t).astype(int)
            tp = ((pred == 1) & (yv == 1)).sum(); fp = ((pred == 1) & (yv == 0)).sum(); fn = ((pred == 0) & (yv == 1)).sum()
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f = 2 * pr * rc / max(pr + rc, 1e-9)
            if f > bf:
                bf, bt = f, t
        return bt
    t_abs = pick(pv); zt = pick((pv - pm) / psd)

    def f1(cases, fn):
        eh = et = ch = ct = 0
        for lab, F in cases:
            pred = fn(F[:, Xtr_cols])
            if lab == -1:
                ct += 1; ch += (pred == -1)
            else:
                et += 1; eh += (pred == lab)
        ae = eh / max(et, 1); ac = ch / max(ct, 1)
        return 2 * ae * ac / max(ae + ac, 1e-9) * 100
    out = {}
    for D in EVAL_DIRS:
        cases = get_eval(D)
        allF = np.concatenate([F[:, Xtr_cols] for _, F in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
        A = f1(cases, lambda F: next((i for i, p in enumerate(prob((F - mu) / sd)) if p > t_abs), -1))
        B = f1(cases, lambda F: next((i for i, p in enumerate(prob((F - mud) / sdd)) if p > t_abs), -1))
        BC = f1(cases, lambda F: next((i for i, z in enumerate((prob((F - mud) / sdd) - pm) / psd) if z > zt), -1))
        out[D] = (A, B, BC)
    print(f"  [dim={len(Xtr_cols)}] val-AUC={vauc:.3f} t_abs={t_abs:.2f}", flush=True)
    return out


mid_cols = list(range(MID * Hsub, (MID + 1) * Hsub))
all_cols = list(range(Dtot))
r9 = evaluate(mid_cols)
rc = evaluate(all_cols)
print(f"\n{'dataset':16s} {'-9only A/B/BC':>18} {'concat A/B/BC':>18}")
for D in EVAL_DIRS:
    print(f"{D:16s}  {r9[D][0]:4.1f}/{r9[D][1]:4.1f}/{r9[D][2]:<4.1f}   {rc[D][0]:4.1f}/{rc[D][1]:4.1f}/{rc[D][2]:<4.1f}", flush=True)
