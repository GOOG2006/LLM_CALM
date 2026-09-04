"""1.5B: dump fthead-1.5B 的 P_A + label 到 npz(全量 ProcessBench)。之后纯numpy跑校准策略。
基座=DSR1-1.5B, 头=fthead_lora + fthead_heads.pt。eval 数据=pb_*_full_in/*/sample.json。
用法: python gen15_dump_pa.py pa15_full.npz
"""
import json, glob, os, sys, re, torch, numpy as np
import torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = 'models/DSR1-1.5B'; L9, MAXLEN = -9, 3072
OUT = sys.argv[1] if len(sys.argv) > 1 else 'pa15_full.npz'
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(base, 'fthead_lora').eval()
H = base.config.hidden_size
head_last = nn.Linear(H, 1).cuda().float(); head_9 = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, 1)).cuda().float()
hd = torch.load('fthead_heads.pt', map_location='cuda')
head_last.load_state_dict(hd['head_last']); head_9.load_state_dict(hd['head_9']); head_last.eval(); head_9.eval()
print("loaded fthead-1.5B", flush=True)


def build(problem, steps):
    steps = [s for s in steps if isinstance(s, str) and s]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    ids = list(tok(problem + "\n", add_special_tokens=True).input_ids); bd = []
    for s in steps:
        a = len(ids); ids += tok(s + "\n", add_special_tokens=False).input_ids; bd.append((a, len(ids)))
    if len(ids) > MAXLEN:
        ids = ids[:MAXLEN]; bd = [(a, min(b, MAXLEN)) for a, b in bd if a < MAXLEN]
    return ids, bd


def PA_of(problem, steps):
    ids, bd = build(problem, steps)
    if not bd:
        return None
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'), output_hidden_states=True)
        h9 = out.hidden_states[L9][0].float(); hl = out.hidden_states[-1][0].float()
        idxs = [b - 1 for (a, b) in bd]
        lg = head_last(hl[idxs]).squeeze(-1) + head_9(h9[idxs]).squeeze(-1)
        return torch.sigmoid(lg).cpu().numpy(), len(bd)


DOMS = [('gsm8k', 'pb_gsm8k_full_in'), ('math', 'pb_math_full_in'), ('olympiad', 'pb_olympiadbench_full_in'), ('omnimath', 'pb_omnimath_full_in')]
save = {}
for name, d in DOMS:
    PAs, labs = [], []
    for fo in sorted(glob.glob(os.path.join(d, '*'))):
        sp = os.path.join(fo, 'sample.json')
        if not os.path.exists(sp):
            continue
        r = json.load(open(sp)); res = PA_of(r['problem'], r['steps'])
        if res is None:
            continue
        pa, n = res
        lab = r['label'] if (r.get('label', -1) == -1 or r.get('label', -1) < n) else -1
        PAs.append(pa); labs.append(lab)
    save[name + '_PA'] = np.array(PAs, dtype=object); save[name + '_lab'] = np.array(labs)
    print(f"{name}: {len(labs)} cases", flush=True)

# math-val P_A for unsupervised threshold reference (same held-out slice as training)
tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
idx = np.random.default_rng(0).permutation(len(tbl))[:8000][-800:]
vPA, vfe = [], []
for i in idx:
    users = [m['content'] for m in tbl[int(i)]['conversations'] if m['role'] == 'user']
    asts = [m['content'] for m in tbl[int(i)]['conversations'] if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    lab = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0]); problem, s0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    steps = [s0] + users[1:n]
    res = PA_of(problem, steps)
    if res is None or res[1] != len(lab):
        continue
    vPA.append(res[0]); vfe.append(next((j for j, v in enumerate(lab) if v == 1), -1))
save['val_PA'] = np.array(vPA, dtype=object); save['val_fe'] = np.array(vfe)
np.savez(OUT, **save)
print(f"saved {OUT} (val {len(vPA)})", flush=True)
