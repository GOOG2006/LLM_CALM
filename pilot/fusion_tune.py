"""诚实融合: 在 mathval(held-out)调 (a,bias), 固定应用到测试切片。A=残差头, B=GenPRM。
score = a*(logit(P_A)-logit(tA)) + (1-a)*logit(1-P_B); > bias 判错。
"""
import json, glob, os, math, torch, numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = 'models/DSR1-1.5B'; L9, MAXLEN, tA = -9, 4096, 0.92
VAL = 'pb_mathval_out'
TEST = [('gsm8k', 'pb_gsm8k_best120_out', 52.8), ('math', 'pb_math_dv_out', 66.6),
        ('olympiad', 'pb_olympiadbench_dv_out', 55.1), ('omnimath', 'pb_omnimath_dv_out', 54.5)]
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
model = PeftModel.from_pretrained(base, 'fthead_lora').eval()
H = base.config.hidden_size
hl_ = nn.Linear(H, 1).cuda().float(); h9_ = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, 1)).cuda().float()
sd = torch.load('fthead_heads.pt', weights_only=True); hl_.load_state_dict(sd['head_last']); h9_.load_state_dict(sd['head_9'])
hl_.eval(); h9_.eval()
LT = math.log(tA / (1 - tA))


def PA(problem, steps):
    steps = [s for s in steps if isinstance(s, str) and s]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    ids = list(tok(problem + "\n", add_special_tokens=True).input_ids); bd = []
    for s in steps:
        a = len(ids); ids += tok(s + "\n", add_special_tokens=False).input_ids; bd.append((a, len(ids)))
    if len(ids) > MAXLEN:
        ids = ids[:MAXLEN]; bd = [(a, min(b, MAXLEN)) for a, b in bd if a < MAXLEN]
    if not bd:
        return None
    with torch.no_grad():
        o = model(torch.tensor([ids], device='cuda'), output_hidden_states=True)
        idx = [b - 1 for a, b in bd]
        lg = hl_(o.hidden_states[-1][0].float()[idx]).squeeze(-1) + h9_(o.hidden_states[L9][0].float()[idx]).squeeze(-1)
        return torch.sigmoid(lg).cpu().numpy()


def lz(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def load(GEN):
    cases = []
    for fo in sorted(glob.glob(os.path.join(GEN, '*_analyze'))):
        r = json.load(open(os.path.join(fo, 'result_1.json')))
        pa = PA(r['problem'], r['steps'])
        if pa is None or len(pa) != len(r['value']):
            continue
        cases.append((r['label'], lz(pa) - LT, lz(1 - np.array(r['value']))))
    return cases


def f1(cases, a, bias):
    eh = et = ch = ct = 0
    for lab, LA, LB in cases:
        pred = next((i for i in range(len(LA)) if a * LA[i] + (1 - a) * LB[i] > bias), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


def genf1(cases):    # B only
    eh = et = ch = ct = 0
    for lab, LA, LB in cases:
        pred = next((i for i in range(len(LB)) if LB[i] > 0), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


val = load(VAL)
best = (0, 0.5, 0.0)
for a in np.arange(0, 1.01, 0.1):
    for bias in np.arange(-2, 2.01, 0.25):
        f = f1(val, a, bias)
        if f > best[0]:
            best = (f, round(a, 1), round(bias, 2))
_, A, B = best
print(f"tuned on mathval: a={A} bias={B} val-F1={best[0]:.1f}\n", flush=True)
print(f"{'dom':10s} {'GenPRM':>7} {'Fusion':>7} {'delta':>6}")
gs = []; fus = []
for name, GEN, ref in TEST:
    cs = load(GEN)
    g = genf1(cs); f = f1(cs, A, B); gs.append(g); fus.append(f)
    print(f"{name:10s} {g:7.1f} {f:7.1f} {f-g:+6.1f}", flush=True)
print(f"{'MEAN':10s} {np.mean(gs):7.1f} {np.mean(fus):7.1f} {np.mean(fus)-np.mean(gs):+6.1f}", flush=True)
