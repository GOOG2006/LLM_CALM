"""混合验证器原型(gsm8k, 离线): 探针打底 + 在'该code'步改用GenPRM reward。
探针=冻结基座层-9线性(从feats_train_convs.pkl训); GenPRM reward/used_code 从 best120_out 读。
对比: 探针-only / GenPRM-only / 混合(路由门控) / 混合(不确定度门控) + 代码使用比例。
"""
import json, glob, os, re, pickle, torch, numpy as np

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 4096
GEN_DIR = 'pb_gsm8k_best120_out'
_NUM = r'\d[\d,]*\.?\d*'
_COMPUTE = re.compile(_NUM + r'\s*[\+\-\*/×·÷]\s*' + _NUM + r'.{0,25}?=\s*' + _NUM)


def route(t):
    return bool(_COMPUTE.search(t or ''))


from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()


def feats(problem, steps):
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
        return None
    t = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        hs = model(t, output_hidden_states=True).hidden_states[LAYER][0]
        return torch.stack([hs[b - 1] for (a, b) in bounds]).float().cpu().numpy()


# ---- train probe from cached MATH-train features (layer -9) ----
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


# threshold on MATH val (step-F1 optimal)
pv = prob(Xz[va]); yv = y[va]
bt, bf = 0.5, -1
for t in np.arange(-3, 3.01, 0.1):
    pr = (pv > t).astype(int)
    tp = ((pr == 1) & (yv == 1)).sum(); fp = ((pr == 1) & (yv == 0)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
    if f > bf:
        bf, bt = f, t
t_abs = bt
print(f"probe trained; t_abs={t_abs:.2f}", flush=True)

# ---- load gsm8k cases: probe feats + GenPRM reward/used_code ----
cases = []   # dict per case
for fo in sorted(glob.glob(os.path.join(GEN_DIR, '*_analyze'))):
    r = json.load(open(os.path.join(fo, 'result_1.json')))
    steps = [s for s in r['steps'] if s is not None]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    F = feats(r['problem'], steps)
    if F is None or F.shape[0] != len(r['value']):
        continue
    cases.append(dict(label=r['label'], F=F, val=np.array(r['value']),
                      uc=r.get('used_code', [route(s) for s in steps]), steps=steps))

# per-domain norm for probe on gsm8k
allF = np.concatenate([c['F'] for c in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
for c in cases:
    c['pP'] = prob((c['F'] - mud) / sdd)          # probe P(error), domain-normalized


def f1(pred_of):
    eh = et = ch = ct = 0
    used = tot = 0
    for c in cases:
        pred, ncode = pred_of(c)
        used += ncode; tot += len(c['val'])
        if c['label'] == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == c['label'])
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100, used / max(tot, 1) * 100


def probe_only(c):
    return next((i for i in range(len(c['pP'])) if c['pP'][i] > t_abs), -1), 0


def genprm_only(c):
    return next((i for i in range(len(c['val'])) if c['val'][i] < 0.5), -1), len(c['val'])


def hybrid_gate(gatefn):
    def f(c):
        g = gatefn(c); nc = int(sum(g))
        err = [(c['val'][i] < 0.5) if g[i] else (c['pP'][i] > t_abs) for i in range(len(c['val']))]
        return (next((i for i, e in enumerate(err) if e), -1), nc)
    return f


gate_router = lambda c: [route(s) for s in c['steps']]
gate_usedcode = lambda c: [bool(x) for x in c['uc']]
def gate_uncertain(margin):
    return lambda c: [abs(c['pP'][i] - t_abs) < margin for i in range(len(c['pP']))]


print(f"\n{'policy':28s} {'F1':>6} {'code%':>6}")
for name, fn in [
    ('probe-only', probe_only),
    ('genprm-only', genprm_only),
    ('hybrid(router-regex)', hybrid_gate(gate_router)),
    ('hybrid(genprm-used_code)', hybrid_gate(gate_usedcode)),
    ('hybrid(uncertain m=.15)', hybrid_gate(gate_uncertain(0.15))),
    ('hybrid(uncertain m=.25)', hybrid_gate(gate_uncertain(0.25))),
    ('hybrid(uncertain m=.35)', hybrid_gate(gate_uncertain(0.35))),
]:
    a, cf = f1(fn)
    print(f"{name:28s} {a:6.1f} {cf:6.1f}", flush=True)
