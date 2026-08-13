"""诊断 math 短板: 残差头在 math 全量上 accE/accC, 按解答长度/错误步位置分箱看漏错分布。"""
import json, glob, os, torch, numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = 'models/DSR1-1.5B'; L9, MAXLEN = -9, 2048
THR = 0.92
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
model = PeftModel.from_pretrained(base, 'fthead_lora').eval()
H = base.config.hidden_size
head_last = nn.Linear(H, 1).cuda().float(); head_9 = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, 1)).cuda().float()
sd = torch.load('fthead_heads.pt'); head_last.load_state_dict(sd['head_last']); head_9.load_state_dict(sd['head_9'])
head_last.eval(); head_9.eval()


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
        return torch.sigmoid(lg).cpu().numpy(), len(bounds)


rows = []
for f in sorted(glob.glob('pb_math_full_in/*')):
    if not os.path.isdir(f):
        continue
    d = json.load(open(os.path.join(f, 'sample.json')))
    r = P(d['problem'], d['steps'])
    if r is None:
        continue
    p, nst = r
    lab = d.get('label', -1)
    if lab is not None and lab >= nst:
        lab = -1
    pred = next((i for i, x in enumerate(p) if x > THR), -1)
    rows.append((lab, pred, nst))

# overall
et = sum(1 for l, _, _ in rows if l != -1); eh = sum(1 for l, pr, _ in rows if l != -1 and pr == l)
ct = sum(1 for l, _, _ in rows if l == -1); ch = sum(1 for l, pr, _ in rows if l == -1 and pr == -1)
ae = eh / max(et, 1); ac = ch / max(ct, 1)
print(f"math full n={len(rows)}  accE={ae*100:.1f}({eh}/{et})  accC={ac*100:.1f}({ch}/{ct})  F1={2*ae*ac/max(ae+ac,1e-9)*100:.1f}")

# recall by solution length (error cases only)
print("\naccE by n_steps bin (error cases):")
for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 999)]:
    sub = [(l, pr) for l, pr, nst in rows if l != -1 and lo <= nst <= hi]
    if sub:
        hit = sum(1 for l, pr in sub if pr == l)
        print(f"  steps {lo}-{hi if hi < 999 else '+'}: accE={hit/len(sub)*100:.1f}% (n={len(sub)})")

# recall by error-step position (early vs late fraction)
print("\naccE by error position (fraction into solution):")
for lo, hi in [(0.0, 0.34), (0.34, 0.67), (0.67, 1.01)]:
    sub = [(l, pr) for l, pr, nst in rows if l != -1 and lo <= l / max(nst, 1) < hi]
    if sub:
        hit = sum(1 for l, pr in sub if pr == l)
        print(f"  pos {lo:.2f}-{hi:.2f}: accE={hit/len(sub)*100:.1f}% (n={len(sub)})")
