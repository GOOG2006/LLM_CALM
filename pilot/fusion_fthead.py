"""融合: 强残差头(A) + GenPRM(B)。logit空间对齐各自边界再加权。
score = a*(logit(P_A)-logit(tA)) + (1-a)*logit(1-P_B); >0 判错。
A=残差头(fthead_lora), P_B 从 GenPRM 输出。120切片。
"""
import json, glob, os, math, torch, numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = 'models/DSR1-1.5B'; L9, MAXLEN, tA = -9, 4096, 0.92
DOMS = [('gsm8k', 'pb_gsm8k_best120_out'), ('math', 'pb_math_dv_out'),
        ('olympiad', 'pb_olympiadbench_dv_out'), ('omnimath', 'pb_omnimath_dv_out')]
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
    ids = list(tok(problem + "\n", add_special_tokens=True).input_ids)
    bd = []
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


def f1_of(cases, decide):
    eh = et = ch = ct = 0
    for lab, LA, LB in cases:
        pred = decide(LA, LB)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


print(f"{'dom':10s} {'A-head':>7} {'B-gen':>7} {'fuse.5':>7} {'oracle':>7}(a,bias)")
allcases = {}
for name, GEN in DOMS:
    cases = []
    for fo in sorted(glob.glob(os.path.join(GEN, '*_analyze'))):
        r = json.load(open(os.path.join(fo, 'result_1.json')))
        pa = PA(r['problem'], r['steps'])
        if pa is None or len(pa) != len(r['value']):
            continue
        LA = lz(pa) - LT               # A error-evidence (0 at boundary tA)
        LB = lz(1 - np.array(r['value']))   # B error-evidence (0 at PB=0.5)
        cases.append((r['label'], LA, LB))
    allcases[name] = cases
    A = f1_of(cases, lambda LA, LB: next((i for i in range(len(LA)) if LA[i] > 0), -1))
    B = f1_of(cases, lambda LA, LB: next((i for i in range(len(LB)) if LB[i] > 0), -1))
    F = f1_of(cases, lambda LA, LB: next((i for i in range(len(LA)) if 0.5 * LA[i] + 0.5 * LB[i] > 0), -1))
    best = (0, None)
    for a in np.arange(0, 1.01, 0.1):
        for bias in np.arange(-2, 2.01, 0.25):
            f = f1_of(cases, lambda LA, LB, a=a, bi=bias: next((i for i in range(len(LA)) if a * LA[i] + (1 - a) * LB[i] > bi), -1))
            if f > best[0]:
                best = (f, (round(a, 1), round(bias, 2)))
    print(f"{name:10s} {A:7.1f} {B:7.1f} {F:7.1f} {best[0]:7.1f}{best[1]}", flush=True)
