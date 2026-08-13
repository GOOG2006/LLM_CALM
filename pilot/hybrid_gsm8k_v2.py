"""混合验证器(gsm8k, 真实版): 探针 + 在探针不确定的步用'基座+代码'reward。
对比: 探针-only / GenPRM-only(参考) / basecode-only / 混合(GenPRM上界) / 混合(基座真实)。
GenPRM reward 从 pb_gsm8k_best120_out; 基座+代码 reward 从 pb_gsm8k_basecode_out。
"""
import json, glob, os, re, pickle, torch, numpy as np

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 4096
GEN_DIR, BASE_DIR = 'pb_gsm8k_best120_out', 'pb_gsm8k_basecode_out'

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


# ---- train probe ----
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


pv = prob(Xz[va]); yv = y[va]
bt, bf = 0.5, -1
for t in np.arange(-3, 3.01, 0.1):
    prb = (pv > t).astype(int)
    tp = ((prb == 1) & (yv == 1)).sum(); fp = ((prb == 1) & (yv == 0)).sum(); fn = ((prb == 0) & (yv == 1)).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
    if f > bf:
        bf, bt = f, t
t_abs = bt
print(f"t_abs={t_abs:.2f}", flush=True)

# ---- base rewards by folder name ----
base_val = {}
for fo in glob.glob(os.path.join(BASE_DIR, '*_analyze')):
    nm = os.path.basename(fo).replace('_analyze', '')
    p = os.path.join(fo, 'result_1.json')
    if os.path.exists(p):
        base_val[nm] = np.array(json.load(open(p))['value'])

cases = []
for fo in sorted(glob.glob(os.path.join(GEN_DIR, '*_analyze'))):
    nm = os.path.basename(fo).replace('_analyze', '')
    r = json.load(open(os.path.join(fo, 'result_1.json')))
    steps = [s for s in r['steps'] if s is not None]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    F = feats(r['problem'], steps)
    if F is None or F.shape[0] != len(r['value']) or nm not in base_val or len(base_val[nm]) != len(r['value']):
        continue
    cases.append(dict(nm=nm, label=r['label'], F=F, gval=np.array(r['value']), bval=base_val[nm]))

allF = np.concatenate([c['F'] for c in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
for c in cases:
    c['pP'] = prob((c['F'] - mud) / sdd)
print(f"aligned cases={len(cases)}", flush=True)


def f1(pred_of):
    eh = et = ch = ct = 0; used = tot = 0
    for c in cases:
        pred, nc = pred_of(c)
        used += nc; tot += len(c['gval'])
        if c['label'] == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == c['label'])
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100, used / max(tot, 1) * 100


def hyb(valkey, margin):
    def f(c):
        n = len(c['gval']); nc = 0; err = []
        for i in range(n):
            gate = abs(c['pP'][i] - t_abs) < margin
            if gate:
                nc += 1; err.append(c[valkey][i] < 0.5)
            else:
                err.append(c['pP'][i] > t_abs)
        return next((i for i, e in enumerate(err) if e), -1), nc
    return f


print(f"\n{'policy':30s} {'F1':>6} {'code%':>6}")
print("%-30s %6.1f %6.1f" % ('probe-only', *f1(lambda c: (next((i for i in range(len(c['pP'])) if c['pP'][i] > t_abs), -1), 0))))
print("%-30s %6.1f %6.1f" % ('genprm-only', *f1(lambda c: (next((i for i in range(len(c['gval'])) if c['gval'][i] < 0.5), -1), len(c['gval'])))))
print("%-30s %6.1f %6.1f" % ('basecode-only', *f1(lambda c: (next((i for i in range(len(c['bval'])) if c['bval'][i] < 0.5), -1), len(c['bval'])))), flush=True)
for m in [0.15, 0.25, 0.35]:
    print("%-30s %6.1f %6.1f" % (f'hybrid GENPRM m={m}', *f1(hyb('gval', m))))
for m in [0.15, 0.25, 0.35]:
    print("%-30s %6.1f %6.1f" % (f'hybrid BASE  m={m}', *f1(hyb('bval', m))), flush=True)
