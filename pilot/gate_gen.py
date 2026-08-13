"""生成探针门控: 对 gsm8k 每案算层-9探针P(域归一化), 门控=不确定(|P-t|<0.4)。
输出 gate_gsm8k.json {folder: [0/1 每步]} 和 probeP_gsm8k.json {folder:[P每步]} + t_abs。
"""
import json, glob, os, pickle, torch, numpy as np

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 4096
DATA = 'pb_gsm8k_120_in'
MARGIN = 0.40

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


# train probe from MATH-train cache
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
    pr = (pv > t).astype(int)
    tp = ((pr == 1) & (yv == 1)).sum(); fp = ((pr == 1) & (yv == 0)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
    if f > bf:
        bf, bt = f, t
t_abs = float(bt)

# per-case feats
raw = {}
for fo in sorted(glob.glob(os.path.join(DATA, '*'))):
    if not os.path.isdir(fo):
        continue
    d = json.load(open(os.path.join(fo, 'sample.json')))
    F = feats(d['problem'], d['steps'])
    if F is None:
        continue
    raw[os.path.basename(fo)] = F
allF = np.concatenate(list(raw.values())); mud, sdd = allF.mean(0), allF.std(0) + 1e-6

gate, probeP = {}, {}
ncode = ntot = 0
for nm, F in raw.items():
    P = prob((F - mud) / sdd)
    probeP[nm] = [float(x) for x in P]
    g = [int(abs(P[i] - t_abs) < MARGIN) for i in range(len(P))]
    gate[nm] = g; ncode += sum(g); ntot += len(g)
json.dump(gate, open('gate_gsm8k.json', 'w'))
json.dump({'t_abs': t_abs, 'margin': MARGIN, 'probeP': probeP}, open('probeP_gsm8k.json', 'w'))
print(f"t_abs={t_abs:.2f} margin={MARGIN} cases={len(gate)} gated_steps={ncode}/{ntot}={ncode/ntot*100:.1f}%")
