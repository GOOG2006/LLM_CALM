"""JEPA 残差验证器(纯自监督, 推理零标签): 残差_i = ||predictor(前缀表示) - 实际步表示||。
高残差 = 从前提推不出 = 错误。测阈值无关诊断 + oracle上界F1。
用法: python jepa_eval.py <data_dir> [limit]
"""
import json, glob, os, sys, math, torch, numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 2048
DATA = sys.argv[1] if len(sys.argv) > 1 else 'pb_math_120_in'
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
model = PeftModel.from_pretrained(base, 'jepa_lora').eval()
H = base.config.hidden_size
predictor = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H)).cuda().float()
predictor.load_state_dict(torch.load('jepa_predictor.pt')); predictor.eval()


def build(problem, steps):
    steps = [s for s in steps if isinstance(s, str) and s]
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


def residuals(ids, bounds):
    t = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        hs_on = model(t, output_hidden_states=True).hidden_states[LAYER][0].float()
        with model.disable_adapter():
            hs_tg = model(t, output_hidden_states=True).hidden_states[LAYER][0].float()
        res = [0.0]                                   # step 0 has no prefix-in-solution; neutral
        for (a, b) in bounds[1:]:
            pred = predictor(hs_on[a - 1:a])
            z = hs_tg[a:b].mean(0, keepdim=True)
            res.append(float((pred - z).pow(2).mean()))
    return res


folders = [f for f in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(f)][:LIMIT]
records = []
for f in folders:
    d = json.load(open(os.path.join(f, 'sample.json')))
    ids, bounds = build(d['problem'], d['steps'])
    if len(bounds) < 2:
        continue
    lab = d.get('label', -1)
    if lab is not None and lab >= len(bounds):
        lab = -1
    records.append((lab, len(bounds), residuals(ids, bounds)))


def z(xs):
    m = sum(xs) / len(xs); sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs), 1)) or 1e-9
    return [(x - m) / sd for x in xs]


top1 = nerr = 0; pct = chance = 0.0
for lab, nst, r in records:
    if lab is None or lab < 0 or nst < 2:
        continue
    nerr += 1
    amax = max(range(nst), key=lambda i: r[i])
    top1 += (amax == lab)
    pct += sum(1 for x in r if x <= r[lab]) / nst
    chance += 1.0 / nst
print(f"[{DATA}] cases={len(records)} err={nerr}")
if nerr:
    print(f"  top1-localization={top1/nerr*100:.1f}% (chance {chance/nerr*100:.1f}%)  mean-percentile={pct/nerr*100:.1f}% (chance 50)")
best = (0.0, None)
for tau in [x / 10 for x in range(-5, 31)]:
    eh = et = ch = ct = 0
    for lab, nst, r in records:
        zz = z(r) if nst > 1 else [0.0]
        pred = next((i for i, v in enumerate(zz) if v > tau), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); f1 = 2 * ae * ac / max(ae + ac, 1e-9)
    if f1 > best[0]:
        best = (f1, (tau, ae, ac))
if best[1]:
    tau, ae, ac = best[1]
    print(f"  oracle-thr F1={best[0]*100:.1f} (tau={tau:.1f} accE={ae*100:.0f} accC={ac*100:.0f})")
