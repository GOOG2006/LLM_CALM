"""诚实调参: 在 MATH-train held-out(pb_mathval_out)上调 α/阈值, 固定应用到4测试子集。
分数 = α*P_A + (1-α)*(1-P_B); 首错=第一个分数>th的步。对比 GenPRM(B) vs 双通路(tuned)。
"""
import json, glob, os, pickle, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'models/DSR1-1.5B'; LAYER, MAXLEN = -9, 4096
VAL = 'pb_mathval_out'
TEST = [('gsm8k', 'pb_gsm8k_best120_out'), ('math', 'pb_math_dv_out'),
        ('olympiad', 'pb_olympiadbench_dv_out'), ('omnimath', 'pb_omnimath_dv_out')]

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()


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
        hs = mdl(t, output_hidden_states=True).hidden_states[LAYER][0]
        return torch.stack([hs[b - 1] for (a, b) in bounds]).float().cpu().numpy()


convs = [c for c in pickle.load(open('feats_train_convs.pkl', 'rb')) if c is not None]
X = np.concatenate([F for _, F in convs]); y = np.concatenate([lab for lab, _ in convs])
mu = X.mean(0); sd = X.std(0) + 1e-6; H = X.shape[1]
Xt = torch.tensor(((X - mu) / sd).astype(np.float32), device='cuda'); yt = torch.tensor(y, device='cuda')
w = torch.zeros(H, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.05)
lf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y.mean()) / max(y.mean(), 1e-3), device='cuda'))
for _ in range(600):
    opt.zero_grad(); loss = lf(Xt @ w + b, yt) + 1e-2 * (w * w).sum(); loss.backward(); opt.step()
w = w.detach(); b = b.detach()


def prob(Fz):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(Fz, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()


def load_dom(GEN):
    cases = []
    for fo in sorted(glob.glob(os.path.join(GEN, '*_analyze'))):
        r = json.load(open(os.path.join(fo, 'result_1.json')))
        F = feats(r['problem'], r['steps'])
        if F is None or F.shape[0] != len(r['value']):
            continue
        cases.append((r['label'], F, np.array(r['value'])))
    allF = np.concatenate([F for _, F, _ in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
    return [(lab, prob((F - mud) / sdd), PB) for lab, F, PB in cases]


def loc_f1(cases, al, th):
    eh = et = ch = ct = 0
    for lab, PA, PB in cases:
        sc = [al * PA[i] + (1 - al) * (1 - PB[i]) for i in range(len(PA))]
        pred = next((i for i, s in enumerate(sc) if s > th), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


def genprm_f1(cases):
    eh = et = ch = ct = 0
    for lab, PA, PB in cases:
        pred = next((i for i, v in enumerate(PB) if v < 0.5), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


val = load_dom(VAL)
best = (0, 0.5, 0.5)
for al in np.arange(0.0, 1.01, 0.1):
    for th in np.arange(0.2, 0.81, 0.025):
        f = loc_f1(val, al, th)
        if f > best[0]:
            best = (f, round(al, 2), round(th, 3))
_, AL, TH = best
print(f"tuned on mathval(n={len(val)}): alpha={AL} th={TH} val-F1={best[0]:.1f}", flush=True)

print(f"\n{'dataset':12s} {'GenPRM':>8} {'DualView':>9} {'delta':>7}")
gv = []; dv = []
for name, GEN in TEST:
    cs = load_dom(GEN)
    g = genprm_f1(cs); d = loc_f1(cs, AL, TH)
    gv.append(g); dv.append(d)
    print(f"{name:12s} {g:8.1f} {d:9.1f} {d-g:+7.1f}", flush=True)
print(f"{'MEAN':12s} {np.mean(gv):8.1f} {np.mean(dv):9.1f} {np.mean(dv)-np.mean(gv):+7.1f}", flush=True)
