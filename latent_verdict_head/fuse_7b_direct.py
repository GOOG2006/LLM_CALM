"""独立融合: 用指定 fthead-7B checkpoint 的 P_A 与 GenPRM-7B P_B 融合。
报 fthead / GenPRM / fuse.5 / fuseOR(在该集上扫最优=上界) + 留一域外诚实融合(在另3域调,用到第4域)。
用法: python gen7b_fuse_ckpt.py ck7b_4000.pt
"""
import json, glob, os, sys, re, torch, numpy as np
import torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

MODEL = 'models/DSR1-7B'; L9, MAXLEN = -9, 1792
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'ck7b_4000.pt'
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")).eval()
H = model.config.hidden_size
head_last = nn.Linear(H, 1).cuda().float(); head_9 = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, 1)).cuda().float()
ck = torch.load(CKPT, map_location='cuda')
set_peft_model_state_dict(model, ck['lora']); head_last.load_state_dict(ck['head_last']); head_9.load_state_dict(ck['head_9'])
head_last.eval(); head_9.eval()
print(f"loaded {CKPT} (step {ck.get('step','?')})", flush=True)


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


def logits_of(ids, bd):
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'), output_hidden_states=True)
        h9 = out.hidden_states[L9][0].float(); hl = out.hidden_states[-1][0].float()
        idxs = [b - 1 for (a, b) in bd]
        return head_last(hl[idxs]).squeeze(-1) + head_9(h9[idxs]).squeeze(-1)


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']; asts = [m['content'] for m in conv if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    lab = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0]); problem, s0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    return problem, [s0] + users[1:n], lab[:n]


# ---- tune fthead threshold tA on GenPRM-Data held-out ----
tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
idx = np.random.default_rng(0).permutation(len(tbl))[:8000][-800:]
val = []
for i in idx:
    p, s, lab = parse_conv(tbl[int(i)]['conversations']); ids, bd = build(p, s)
    if bd and len(bd) == len(lab):
        val.append((next((j for j, v in enumerate(lab) if v == 1), -1), torch.sigmoid(logits_of(ids, bd)).cpu().numpy()))


def locf1(cases, t):
    eh = et = ch = ct = 0
    for lab, P in cases:
        pred = next((i for i, p in enumerate(P) if p > t), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); return 2 * ae * ac / max(ae + ac, 1e-9) * 100


tA, bf = 0.5, -1
for t in np.arange(0.30, 0.85, 0.02):
    f = locf1(val, t)
    if f > bf:
        bf, tA = f, t
print(f"tuned tA={tA:.2f}", flush=True)

OUTS = [('gsm8k', 'pb_gsm8k_7b_out'), ('math', 'pb_math_7b_out'), ('olympiad', 'pb_olympiadbench_7b_out'), ('omnimath', 'pb_omnimath_7b_out')]
REF = {'gsm8k': 83.4, 'math': 80.0, 'olympiad': 72.3, 'omnimath': 71.5}


def cases_of(OUT):
    cs = []
    for fo in sorted(glob.glob(os.path.join(OUT, '*_analyze'))):
        rp = os.path.join(fo, 'result_1.json')
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp)); ids, bd = build(r['problem'], r['steps'])
        if not bd:
            continue
        PA = torch.sigmoid(logits_of(ids, bd)).cpu().numpy(); PB = np.array(r['value'], float)
        lab = r['label'] if (r['label'] == -1 or r['label'] < len(bd)) else -1
        cs.append((lab, PA, PB))
    return cs


DATA = {name: cases_of(out) for name, out in OUTS}


def _lz(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def f1_pred(cases, predfn):
    eh = et = ch = ct = 0
    for lab, PA, PB in cases:
        pred = predfn(PA, PB)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); return 2 * ae * ac / max(ae + ac, 1e-9) * 100


def fused(a, bias):
    return lambda PA, PB: next((i for i in range(min(len(PA), len(PB)))
                                if a * (_lz(PA[i]) - _lz(tA)) + (1 - a) * _lz(1 - PB[i]) > bias), -1)


GRID = [(a, b) for a in np.arange(0, 1.01, 0.1) for b in np.arange(-2, 2.01, 0.25)]
print(f"\n{'dom':10s} {'fthead':>7} {'GenPRM':>7} {'fuse.5':>7} {'fuseOR':>7}", flush=True)
A = B = F5 = ORr = 0
rows = {}
for name, _ in OUTS:
    cs = DATA[name]
    fA = f1_pred(cs, lambda PA, PB: next((i for i in range(len(PA)) if PA[i] > tA), -1))
    fB = f1_pred(cs, lambda PA, PB: next((i for i in range(len(PB)) if PB[i] < 0.5), -1))
    f5 = f1_pred(cs, fused(0.5, 0.0))
    perset = [(f1_pred(cs, fused(a, b)), a, b) for a, b in GRID]
    best = max(perset)[0]
    rows[name] = (fA, fB, f5, best)
    print(f"{name:10s} {fA:7.1f} {fB:7.1f} {f5:7.1f} {best:7.1f}", flush=True)
mA = np.mean([rows[n][0] for n, _ in OUTS]); mB = np.mean([rows[n][1] for n, _ in OUTS])
mF5 = np.mean([rows[n][2] for n, _ in OUTS]); mOR = np.mean([rows[n][3] for n, _ in OUTS])
print(f"{'MEAN':10s} {mA:7.1f} {mB:7.1f} {mF5:7.1f} {mOR:7.1f}", flush=True)

# ---- honest: leave-one-domain-out (tune a,bias on other 3 pooled, apply to held-out domain) ----
print("\n-- honest leave-one-domain-out fusion --", flush=True)
hon = []
for name, _ in OUTS:
    others = [c for n2, _ in OUTS if n2 != name for c in DATA[n2]]
    best_ab, best_f = None, -1
    for a, b in GRID:
        f = f1_pred(others, fused(a, b))
        if f > best_f:
            best_f, best_ab = f, (a, b)
    ft = f1_pred(DATA[name], fused(*best_ab))
    hon.append(ft)
    print(f"{name:10s} fuse={ft:6.1f}  (GenPRM {REF[name]:.1f}, a={best_ab[0]:.1f} bias={best_ab[1]:.2f})", flush=True)
print(f"{'MEAN':10s} fuse={np.mean(hon):6.1f}  (GenPRM {np.mean(list(REF.values())):.1f})", flush=True)
