"""全量 ProcessBench 最终评测: 末token探针 / 均值池化探针 / 置信选集成。
一次前向抽两种特征, 缓存 feats_full_<cfg>.npz。探针从 GenPRM-Data 缓存训练。
"""
import json, glob, os, pickle, torch, numpy as np

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 4096
FULL = ['pb_gsm8k_full_in', 'pb_math_full_in', 'pb_olympiadbench_full_in', 'pb_omnimath_full_in']
_m = None


def get_model():
    global _m, tok
    if _m is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        _m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    return _m


def feats_both(problem, steps):
    m = get_model()
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
    if not bounds:
        return None, None
    t = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        hs = m(t, output_hidden_states=True).hidden_states[LAYER][0]
        lt, mp = [], []
        for (a, b) in bounds:
            lt.append(hs[b - 1].float().cpu().numpy())
            seg = hs[a:b] if b > a else hs[b - 1:b]
            mp.append(seg.float().mean(0).cpu().numpy())
    return np.stack(lt), np.stack(mp)


def get_full(DATA):
    cache = f"feats_full_{os.path.basename(DATA)}.npz"
    if os.path.exists(cache):
        d = np.load(cache); Flt, Fmp, lens, labs = d['Flt'], d['Fmp'], d['lens'], d['labs']
        cases, o = [], 0
        for L, lb in zip(lens, labs):
            cases.append((int(lb), Flt[o:o + L], Fmp[o:o + L])); o += L
        return cases
    cases, LT, MP, lens, labs = [], [], [], [], []
    fs = [x for x in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(x)]
    for k, f in enumerate(fs):
        d = json.load(open(os.path.join(f, 'sample.json')))
        flt, fmp = feats_both(d['problem'], d['steps'])
        if flt is None:
            continue
        lab = d.get('label', -1)
        if lab is not None and lab >= flt.shape[0]:
            lab = -1
        cases.append((lab, flt, fmp)); LT.append(flt); MP.append(fmp); lens.append(flt.shape[0]); labs.append(lab)
        if (k + 1) % 200 == 0:
            print(f"  {os.path.basename(DATA)} {k+1}/{len(fs)}", flush=True)
    np.savez(cache, Flt=np.concatenate(LT), Fmp=np.concatenate(MP), lens=np.array(lens), labs=np.array(labs))
    return cases


def train_probe(cache):
    convs = [c for c in pickle.load(open(cache, 'rb')) if c is not None]
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
    pv = prob(Xz[va]); yv = y[va]
    bt, bf = 0.5, -1
    for t in np.arange(-3, 3.01, 0.1):
        pr = (pv > t).astype(int)
        tp = ((pr == 1) & (yv == 1)).sum(); fp = ((pr == 1) & (yv == 0)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
        if f > bf:
            bf, bt = f, t
    return prob, bt


plt_fn, t_lt = train_probe('feats_train_convs.pkl')
pmp_fn, t_mp = train_probe('feats_train_mp.pkl')


def f1_of(cases, decide):
    eh = et = ch = ct = 0
    for lab, Flt, Fmp in cases:
        pred = decide(Flt, Fmp)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100, ae * 100, ac * 100


print(f"\n{'dataset':16s} {'lastTok':>8} {'meanPool':>9} {'confPick':>9}   (accE/accC of best)")
for D in FULL:
    cases = get_full(D)
    Flt_all = np.concatenate([c[1] for c in cases]); mlt, slt = Flt_all.mean(0), Flt_all.std(0) + 1e-6
    Fmp_all = np.concatenate([c[2] for c in cases]); mmp, smp = Fmp_all.mean(0), Fmp_all.std(0) + 1e-6

    def PL(F): return plt_fn((F - mlt) / slt)
    def PM(F): return pmp_fn((F - mmp) / smp)

    lt = f1_of(cases, lambda Fl, Fm: next((i for i, p in enumerate(PL(Fl)) if p > t_lt), -1))[0]
    mp = f1_of(cases, lambda Fl, Fm: next((i for i, p in enumerate(PM(Fm)) if p > t_mp), -1))[0]

    def confpick(Fl, Fm):
        pl = PL(Fl); pm = PM(Fm); m = min(len(pl), len(pm))
        err = [(pl[i] > t_lt) if abs(pl[i] - 0.5) >= abs(pm[i] - 0.5) else (pm[i] > t_mp) for i in range(m)]
        return next((i for i, e in enumerate(err) if e), -1)
    cp, ae, ac = f1_of(cases, confpick)
    print(f"{os.path.basename(D):16s} {lt:8.1f} {mp:9.1f} {cp:9.1f}   ({ae:.0f}/{ac:.0f})", flush=True)
