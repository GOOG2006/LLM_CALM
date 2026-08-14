"""快速判断 7B 基座上哪一层对'步对错'最可分。冻结 DSR1-7B, 取每步末 token 隐状态,
候选层做 case级 train/val 逻辑回归, 报 val-AUC + 定位F1。用法: python layer_sweep_7b.py [nconv]
"""
import re, sys, torch, numpy as np
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s); order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_logreg(X, y, iters=400, C=0.5):
    Xt = torch.tensor(X, device='cuda', dtype=torch.float32); yt = torch.tensor(y, device='cuda', dtype=torch.float32)
    w = torch.zeros(X.shape[1], device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-3), device='cuda')
    opt = torch.optim.Adam([w, b], lr=0.05); lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(iters):
        opt.zero_grad(); loss = lf(Xt @ w + b, yt) + (1.0 / C) * 1e-3 * (w * w).sum(); loss.backward(); opt.step()
    return w.detach(), b.detach()


def decision(X, w, b):
    with torch.no_grad():
        return (torch.tensor(X, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()

MODEL = 'models/DSR1-7B'; MAXLEN = 1792
NCONV = int(sys.argv[1]) if len(sys.argv) > 1 else 700
CAND = [-1, -3, -5, -7, -9, -11, -13, -15]   # of 29 hidden_states (0=embed .. 28=last)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']
    asts = [m['content'] for m in conv if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    lab = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0])
    problem, step0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    return problem, [step0] + users[1:n], lab[:n]


def feats(problem, steps):
    steps = [s for s in steps if isinstance(s, str) and s]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    ids = list(tok(problem + "\n", add_special_tokens=True).input_ids); ends = []
    for s in steps:
        ids += tok(s + "\n", add_special_tokens=False).input_ids; ends.append(len(ids) - 1)
    if len(ids) > MAXLEN:
        ids = ids[:MAXLEN]; ends = [e for e in ends if e < MAXLEN]
    if not ends:
        return None
    with torch.no_grad():
        hs = model(torch.tensor([ids], device='cuda'), output_hidden_states=True).hidden_states
        # [n_cand, n_steps, H]
        return np.stack([torch.stack([hs[L][0][e] for e in ends]).float().cpu().numpy() for L in CAND])


tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
idx = np.random.default_rng(0).permutation(len(tbl))[:NCONV]
cases = []   # (feat[n_cand,ns,H], lab[ns], first_err)
for k, i in enumerate(idx):
    p, s, lab = parse_conv(tbl[int(i)]['conversations'])
    if len(s) < 1 or len(lab) != len(s):
        continue
    F = feats(p, s)
    if F is None or F.shape[1] != len(lab):
        continue
    fe = next((j for j, v in enumerate(lab) if v == 1), -1)
    cases.append((F, np.array(lab), fe))
    if (k + 1) % 200 == 0:
        print(f"  feats {k+1}/{len(idx)}", flush=True)
print(f"cases={len(cases)}", flush=True)

rng = np.random.default_rng(1); perm = rng.permutation(len(cases))
tr, va = perm[:int(len(cases) * 0.8)], perm[int(len(cases) * 0.8):]


def locf1(preds_by_case, va_idx):
    eh = et = ch = ct = 0
    for ci in va_idx:
        _, lab, fe = cases[ci]; P = preds_by_case[ci]
        pred = next((j for j, p in enumerate(P) if p > 0.5), -1)
        if fe == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == fe)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


print(f"\n{'layer(idx)':12s} {'val-AUC':>8} {'locF1@.5':>9}")
results = []
for li, L in enumerate(CAND):
    Xtr = np.concatenate([cases[c][0][li] for c in tr]).astype(np.float32); ytr = np.concatenate([cases[c][1] for c in tr]).astype(np.float32)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    w, b = fit_logreg((Xtr - mu) / sd, ytr)
    Xva = np.concatenate([cases[c][0][li] for c in va]).astype(np.float32); yva = np.concatenate([cases[c][1] for c in va])
    auc = roc_auc(yva, decision((Xva - mu) / sd, w, b))
    preds = {c: 1 / (1 + np.exp(-decision((cases[c][0][li] - mu) / sd, w, b))) for c in va}
    f1 = locf1(preds, va)
    lay = 28 + (L + 1)   # index into human layer number
    print(f"idx{L:>3}(L{lay:<2}) {auc:8.3f} {f1:9.1f}", flush=True)
    results.append((L, auc, f1))
best = max(results, key=lambda r: r[1])
print(f"\nBEST layer idx={best[0]} (AUC={best[1]:.3f}, locF1={best[2]:.1f})  [current head uses -9]", flush=True)
