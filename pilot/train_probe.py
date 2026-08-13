"""方法B v1: 冻结基座 + 层-9 线性探针。
训练数据 = GenPRM 的 MATH-train 对话(每步带 Yes/No 标签, 与ProcessBench不相交)。
在训练集内选固定阈值; 再评 ProcessBench 四子集的首错定位F1(固定阈值, 不在测试集调)。
用法: python train_probe.py [n_train]
"""
import json, glob, os, sys, math, re, torch, numpy as np
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'models/DSR1-1.5B'
LAYER = -9
MAXLEN = 4096
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
EVAL_DIRS = ['pb_gsm8k_120_in', 'pb_math_120_in', 'pb_olympiadbench_120_in', 'pb_omnimath_big_in']

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()


def build(problem, steps):
    steps = [s for s in steps if s is not None]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    full_ids = list(tok(problem + "\n", add_special_tokens=True).input_ids)
    bounds = []
    for s in steps:
        start = len(full_ids)
        full_ids += tok(s + "\n", add_special_tokens=False).input_ids
        bounds.append((start, len(full_ids)))
    if len(full_ids) > MAXLEN:
        full_ids = full_ids[:MAXLEN]
        bounds = [(a, min(b, MAXLEN)) for a, b in bounds if a < MAXLEN]
    return full_ids, bounds


def feats(problem, steps):
    full_ids, bounds = build(problem, steps)
    if len(bounds) < 1:
        return None
    ids = torch.tensor([full_ids], device='cuda')
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states[LAYER][0]
        f = torch.stack([hs[b - 1] for (a, b) in bounds]).float().cpu().numpy()
    return f  # [nst, H]


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']
    asts = [m['content'] for m in conv if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    labels = []
    for a in asts[:n]:
        b = re.findall(r'boxed\{(Yes|No)\}', a)
        labels.append(1 if (b and b[-1] == 'No') else 0)   # No = error
    u0 = re.sub(r'^\s*Question:\s*', '', users[0])
    problem, step0 = (u0.split('\n\n', 1) + [''])[:2] if '\n\n' in u0 else ('', u0)
    steps = [step0] + users[1:n]
    return problem, steps, labels[:len(steps)]


# ---------- build training set ----------
tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
rng = np.random.default_rng(0)
idx = rng.permutation(len(tbl))[:N_TRAIN]
Xtr, ytr = [], []
for k, i in enumerate(idx):
    problem, steps, labels = parse_conv(tbl[int(i)]['conversations'])
    if len(steps) < 1 or len(labels) != len(steps):
        continue
    F = feats(problem, steps)
    if F is None or F.shape[0] != len(labels):
        continue
    Xtr.append(F); ytr.append(np.array(labels, np.float32))
    if (k + 1) % 500 == 0:
        print(f"  train feats {k+1}/{len(idx)}", flush=True)
X = np.concatenate(Xtr); y = np.concatenate(ytr)
print(f"train steps={len(y)}  error_rate={y.mean():.3f}", flush=True)

# hold out 10% for threshold
m = len(y); perm = rng.permutation(m); cut = int(m * 0.9)
tr, va = perm[:cut], perm[cut:]
mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
Xz = (X - mu) / sd
H = X.shape[1]

Xt = torch.tensor(Xz[tr], device='cuda'); yt = torch.tensor(y[tr], device='cuda')
w = torch.zeros(H, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.05)
posw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), device='cuda')
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
for ep in range(600):
    opt.zero_grad()
    loss = lossf(Xt @ w + b, yt) + 1e-2 * (w * w).sum()
    loss.backward(); opt.step()
w = w.detach(); b = b.detach()


def prob(Fz):
    return torch.sigmoid(torch.tensor(Fz, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()


def pick_t(scores, labels):        # scores/labels arrays -> best step-F1 threshold
    bt, bf = 0.5, -1
    for t100 in range(-30, 96, 2):
        t = t100 / 100
        pred = (scores > t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f = 2 * pr * rc / max(pr + rc, 1e-9)
        if f > bf:
            bf, bt = f, t
    return bt, bf


pv = prob(Xz[va]); yv = y[va]
best_t, best_f = pick_t(pv, yv)
print(f"absolute threshold t*={best_t:.2f}  (val step-F1={best_f:.3f})", flush=True)

# for within-case z decision: z-score val probs within each val 'pseudo-case' is unavailable,
# so pick z-threshold on val by treating each val step's global z (using train prob mean/std)
pm, psd = float(pv.mean()), float(pv.std() + 1e-9)
zt, ztf = pick_t((pv - pm) / psd, yv)
print(f"within-case z-threshold zt*={zt:.2f}  (val step-F1={ztf:.3f})", flush=True)

np.savez('probe_dsr1_l9.npz', w=w.cpu().numpy(), b=b.cpu().numpy(), mu=mu, sd=sd,
         best_t=best_t, zt=zt, pm=pm, psd=psd)
print("saved probe_dsr1_l9.npz", flush=True)


# ---------- evaluate on ProcessBench (fixed decisions) ----------
def eval_dir(DATA):
    cases = []
    for f in [x for x in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(x)]:
        d = json.load(open(os.path.join(f, 'sample.json')))
        F = feats(d['problem'], d['steps'])
        if F is None:
            continue
        lab = d.get('label', -1)
        if lab is not None and lab >= F.shape[0]:
            lab = -1
        cases.append((lab, F))
    allF = np.concatenate([F for _, F in cases])
    mud, sdd = allF.mean(0), allF.std(0) + 1e-6           # per-domain unsupervised stats

    def f1(predfn):
        eh = et = ch = ct = 0
        for lab, F in cases:
            pred = predfn(F)
            if lab == -1:
                ct += 1; ch += (pred == -1)
            else:
                et += 1; eh += (pred == lab)
        ae = eh / max(et, 1); ac = ch / max(ct, 1)
        return 2 * ae * ac / max(ae + ac, 1e-9) * 100

    A = f1(lambda F: next((i for i, p in enumerate(prob((F - mu) / sd)) if p > best_t), -1))
    B = f1(lambda F: next((i for i, p in enumerate(prob((F - mud) / sdd)) if p > best_t), -1))

    def cz(F):
        p = prob((F - mu) / sd)
        z = (p - pm) / psd
        return next((i for i, zz in enumerate(z) if zz > zt), -1)
    C = f1(cz)
    return A, B, C


print("\n===== Method B (frozen base DSR1 + layer-9 probe, trained on MATH-train) =====")
print(f"{'dataset':16s} {'A:abs':>7} {'B:domNorm':>10} {'C:caseZ':>8}")
for D in EVAL_DIRS:
    A, B, C = eval_dir(D)
    print(f"{D:16s} {A:7.1f} {B:10.1f} {C:8.1f}", flush=True)
