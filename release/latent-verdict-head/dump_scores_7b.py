"""把指定 fthead-7B ckpt 的 P_A(每步错误概率) + P_B(GenPRM value) + label dump 到 npz。
之后所有校准/融合实验纯 numpy 迭代, 不再占 GPU。用法: python gen7b_dump_pa.py ck7b_4000.pt pa_4000.npz
也 dump math-val 的 P_A(供无标签阈值参考)。
"""
import json, glob, os, sys, re, torch, numpy as np
import torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

MODEL = 'models/DSR1-7B'; L9 = -9
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'ck7b_4000.pt'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'pa_4000.npz'
MAXLEN = int(sys.argv[3]) if len(sys.argv) > 3 else 1792
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")).eval()
H = model.config.hidden_size
head_last = nn.Linear(H, 1).cuda().float(); head_9 = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, 1)).cuda().float()
ck = torch.load(CKPT, map_location='cuda')
set_peft_model_state_dict(model, ck['lora']); head_last.load_state_dict(ck['head_last']); head_9.load_state_dict(ck['head_9'])
head_last.eval(); head_9.eval()
print(f"loaded {CKPT} step {ck.get('step','?')}", flush=True)


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


OUTS = [('gsm8k', 'pb_gsm8k_7b_out'), ('math', 'pb_math_7b_out'), ('olympiad', 'pb_olympiadbench_7b_out'), ('omnimath', 'pb_omnimath_7b_out')]
save = {}
for name, d in OUTS:
    PAs, PBs, labs = [], [], []
    for fo in sorted(glob.glob(os.path.join(d, '*_analyze'))):
        rp = os.path.join(fo, 'result_1.json')
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp)); res = PA_of(r['problem'], r['steps'])
        if res is None:
            continue
        pa, n = res
        lab = r['label'] if (r['label'] == -1 or r['label'] < n) else -1
        PAs.append(pa); PBs.append(np.array(r['value'], float)[:n]); labs.append(lab)
    save[name + '_PA'] = np.array(PAs, dtype=object); save[name + '_PB'] = np.array(PBs, dtype=object); save[name + '_lab'] = np.array(labs)
    print(f"{name}: {len(labs)} cases", flush=True)

# math-val P_A (from GenPRM-Data held-out) for unsupervised threshold reference
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
print(f"saved {OUT} (val {len(vPA)} cases)", flush=True)
