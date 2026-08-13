"""用现有 fthead 模型, 评测 MAXLEN=4096(不重训), 看 math 长解答截断问题能否缓解。全量四子集。"""
import json, glob, os, sys, torch, numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = 'models/DSR1-1.5B'; L9 = -9
MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
THR = 0.92
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
model = PeftModel.from_pretrained(base, 'fthead_lora').eval()
H = base.config.hidden_size
head_last = nn.Linear(H, 1).cuda().float(); head_9 = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, 1)).cuda().float()
sd = torch.load('fthead_heads.pt', weights_only=True); head_last.load_state_dict(sd['head_last']); head_9.load_state_dict(sd['head_9'])
head_last.eval(); head_9.eval()
ref = {'gsm8k': 52.8, 'math': 66.6, 'olympiadbench': 55.1, 'omnimath': 54.5}


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


def P(problem, steps):
    ids, bounds = build(problem, steps)
    if not bounds:
        return None
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'), output_hidden_states=True)
        h9 = out.hidden_states[L9][0].float(); hl = out.hidden_states[-1][0].float()
        idxs = [b - 1 for (a, b) in bounds]
        lg = head_last(hl[idxs]).squeeze(-1) + head_9(h9[idxs]).squeeze(-1)
        return torch.sigmoid(lg).cpu().numpy()


print(f"MAXLEN={MAXLEN}  {'dataset':14s} {'F1':>6} {'accE':>6} {'accC':>6}   GenPRM")
fs = []
for cfg in ['gsm8k', 'math', 'olympiadbench', 'omnimath']:
    eh = et = ch = ct = 0
    for f in sorted(glob.glob(f'pb_{cfg}_full_in/*')):
        if not os.path.isdir(f):
            continue
        d = json.load(open(os.path.join(f, 'sample.json')))
        p = P(d['problem'], d['steps'])
        if p is None:
            continue
        lab = d.get('label', -1)
        if lab is not None and lab >= len(p):
            lab = -1
        pred = next((i for i, x in enumerate(p) if x > THR), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); f1 = 2 * ae * ac / max(ae + ac, 1e-9) * 100
    fs.append(f1)
    print(f"{'':14s} {cfg:14s} {f1:6.1f} {ae*100:6.1f} {ac*100:6.1f}   ({ref[cfg]})", flush=True)
print(f"MEAN F1 = {np.mean(fs):.1f}", flush=True)
