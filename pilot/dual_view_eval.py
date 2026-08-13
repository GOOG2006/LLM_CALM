"""双通路验证器: 通路A(探针,读表示) + 通路B(GenPRM,生成推理)。
测融合能否去相关自洽幻觉、压低证明域假阳。用已有 GenPRM 输出(gsm8k best120 / omnimath crit120)。
探针从 feats_train_convs.pkl 训。逐案抽探针特征与 GenPRM value 对齐。
"""
import json, glob, os, pickle, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'models/DSR1-1.5B'; LAYER, MAXLEN = -9, 4096
DOMS = [('gsm8k', 'pb_gsm8k_best120_out'), ('math', 'pb_math_dv_out'),
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


# train probe
convs = [c for c in pickle.load(open('feats_train_convs.pkl', 'rb')) if c is not None]
X = np.concatenate([F for _, F in convs]); y = np.concatenate([lab for lab, _ in convs])
rng = np.random.default_rng(1); perm = rng.permutation(len(y)); cut = int(len(y) * 0.9); tr, va = perm[:cut], perm[cut:]
mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; H = X.shape[1]
Xt = torch.tensor(((X - mu) / sd)[tr].astype(np.float32), device='cuda'); yt = torch.tensor(y[tr], device='cuda')
w = torch.zeros(H, device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.05)
lf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), device='cuda'))
for _ in range(600):
    opt.zero_grad(); loss = lf(Xt @ w + b, yt) + 1e-2 * (w * w).sum(); loss.backward(); opt.step()
w = w.detach(); b = b.detach()


def prob(Fz):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(Fz, device='cuda', dtype=torch.float32) @ w + b).cpu().numpy()


pv = prob(((X - mu) / sd)[va]); yv = y[va]
bt, bf = 0.5, -1
for t in np.arange(-3, 3.01, 0.05):
    pr = (pv > t).astype(int); tp = ((pr == 1) & (yv == 1)).sum(); fp = ((pr == 1) & (yv == 0)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
    if f > bf:
        bf, bt = f, t
t_A = bt


def metrics(cases, decide):
    eh = et = ch = ct = 0; fp_steps = tot_cor = 0
    for lab, PA, PB in cases:
        # first-error prediction
        err = decide(PA, PB)
        pred = next((i for i, e in enumerate(err) if e), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
        # step-level false positives on truly-correct steps (i<lab, or all if lab==-1)
        for i in range(len(err)):
            correct_step = (lab == -1) or (i < lab)
            if correct_step:
                tot_cor += 1; fp_steps += int(err[i])
    ae = eh / max(et, 1); ac = ch / max(ct, 1); f1 = 2 * ae * ac / max(ae + ac, 1e-9)
    return f1 * 100, ae * 100, ac * 100, fp_steps / max(tot_cor, 1) * 100


for name, GEN in DOMS:
    cases = []
    for fo in sorted(glob.glob(os.path.join(GEN, '*_analyze'))):
        r = json.load(open(os.path.join(fo, 'result_1.json')))
        F = feats(r['problem'], r['steps'])
        if F is None or F.shape[0] != len(r['value']):
            continue
        cases.append((r['label'], F, np.array(r['value'])))
    allF = np.concatenate([F for _, F, _ in cases]); mud, sdd = allF.mean(0), allF.std(0) + 1e-6
    cases = [(lab, prob((F - mud) / sdd), PB) for lab, F, PB in cases]
    print(f"\n===== {name} (n={len(cases)}, t_A={t_A:.2f}) =====")
    fA, aeA, acA, fpA = metrics(cases, lambda PA, PB: [p > t_A for p in PA])
    fB, aeB, acB, fpB = metrics(cases, lambda PA, PB: [v < 0.5 for v in PB])
    print(f"A:probe   F1={fA:.1f}  |  B:genprm F1={fB:.1f} (FP {fpB:.1f})")
    best = (0, None)
    for anc in np.arange(0.05, 0.71, 0.05):
        f1, ae, ac, fp = metrics(cases, lambda PA, PB, a=anc: [(PB[i] < 0.5) and (PA[i] > a) for i in range(len(PA))])
        if f1 > best[0]:
            best = (f1, (anc, ae, ac, fp))
    anc, ae, ac, fp = best[1]
    # soft fusion: score = a*PA + (1-a)*(1-PB); sweep a and threshold
    bestsoft = (0, None)
    for al in np.arange(0.0, 1.01, 0.1):
        sc_cases = [(lab, [al * PA[i] + (1 - al) * (1 - PB[i]) for i in range(len(PA))]) for lab, PA, PB in cases]
        for th in np.arange(0.2, 0.81, 0.05):
            eh = et = ch = ct = 0
            for lab, sc in sc_cases:
                pred = next((i for i, s in enumerate(sc) if s > th), -1)
                if lab == -1:
                    ct += 1; ch += (pred == -1)
                else:
                    et += 1; eh += (pred == lab)
            a2 = eh / max(et, 1); c2 = ch / max(ct, 1); f = 2 * a2 * c2 / max(a2 + c2, 1e-9) * 100
            if f > bestsoft[0]:
                bestsoft = (f, (al, th))
    # fixed (no tuning): alpha=0.5, threshold=0.5
    def fixedf1(al, th):
        eh = et = ch = ct = 0
        for lab, PA, PB in cases:
            sc = [al * PA[i] + (1 - al) * (1 - PB[i]) for i in range(len(PA))]
            pred = next((i for i, s in enumerate(sc) if s > th), -1)
            if lab == -1:
                ct += 1; ch += (pred == -1)
            else:
                et += 1; eh += (pred == lab)
        a2 = eh / max(et, 1); c2 = ch / max(ct, 1); return 2 * a2 * c2 / max(a2 + c2, 1e-9) * 100
    print(f"B-anchored best={best[0]:.1f}  soft-oracle={bestsoft[0]:.1f}(a={bestsoft[1][0]:.1f})  soft-FIXED a=.5,th=.5 -> {fixedf1(0.5, 0.5):.1f}  a=.5,th=.45 -> {fixedf1(0.5, 0.45):.1f}", flush=True)
