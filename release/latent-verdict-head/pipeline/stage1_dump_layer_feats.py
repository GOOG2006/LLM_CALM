"""Dump 逐层步末特征 + 标签, 供'基座自写的选层代码'消费。冻结 DSR1-7B。
train: 每层池化的步特征 Xtr[cand,steps,H] + ytr(步错=1); val: 每案 per-层 per-步特征 + 首错下标。
用法: python dump_layer_feats.py [ntrain] [nval]
"""
import re, sys, torch, numpy as np
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'models/DSR1-7B'; MAXLEN = 1792
CAND = [-1, -3, -5, -7, -9, -11, -13, -15]
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 400
NVA = int(sys.argv[2]) if len(sys.argv) > 2 else 150
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']; asts = [m['content'] for m in conv if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    lab = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0]); problem, s0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    return problem, [s0] + users[1:n], lab[:n]


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
        return np.stack([torch.stack([hs[L][0][e] for e in ends]).float().cpu().numpy() for L in CAND])  # [cand,steps,H]


tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
idx = np.random.default_rng(0).permutation(len(tbl))[:NTR + NVA]
cases = []
for i in idx:
    p, s, lab = parse_conv(tbl[int(i)]['conversations'])
    if len(s) < 1 or len(lab) != len(s):
        continue
    F = feats(p, s)
    if F is None or F.shape[1] != len(lab):
        continue
    fe = next((j for j, v in enumerate(lab) if v == 1), -1)
    cases.append((F.astype(np.float32), np.array(lab, np.int8), fe))
    if len(cases) % 100 == 0:
        print(f"  {len(cases)} cases", flush=True)

tr, va = cases[:NTR], cases[NTR:NTR + NVA]
Xtr = np.concatenate([c[0] for c in tr], axis=1)          # [cand, tot_steps, H]
ytr = np.concatenate([c[1] for c in tr])                  # [tot_steps]
val_feats = np.empty(len(va), dtype=object)
for i, c in enumerate(va):
    val_feats[i] = c[0]                                          # each [cand,steps,H]
np.savez('layerfeats.npz',
         cand=np.array(CAND),
         Xtr=Xtr, ytr=ytr,
         val_feats=val_feats,
         val_fe=np.array([c[2] for c in va]))
print(f"saved layerfeats.npz  cand={CAND}  train_steps={len(ytr)}  val_cases={len(va)}", flush=True)
