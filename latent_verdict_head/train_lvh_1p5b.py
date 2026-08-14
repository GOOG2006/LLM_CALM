"""残差判词头 + LoRA 微调: logit = Linear(h_last) + MLP(h_9^步末)。
在 GenPRM-Data 步级 Yes/No 上训(表示随 LoRA 更新)。测证明域能否 > 冻结探针的 45。
用法: python finetune_head.py [n_train_convs] [nsteps]
"""
import json, glob, os, sys, re, torch, numpy as np
import torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = 'models/DSR1-1.5B'; L9, MAXLEN = -9, 2048
NCONV = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
H = model.config.hidden_size
head_last = nn.Linear(H, 1).cuda().float()
head_9 = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, 1)).cuda().float()


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


def logits_of(ids, bounds, grad):
    t = torch.tensor([ids], device='cuda')
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(t, output_hidden_states=True)
        h9 = out.hidden_states[L9][0].float(); hl = out.hidden_states[-1][0].float()
        idxs = [b - 1 for (a, b) in bounds]
        return head_last(hl[idxs]) .squeeze(-1) + head_9(h9[idxs]).squeeze(-1)


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']
    asts = [m['content'] for m in conv if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    labels = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0])
    problem, step0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    steps = [step0] + users[1:n]
    return problem, steps, labels[:len(steps)]


tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
idx = np.random.default_rng(0).permutation(len(tbl))[:NCONV]
seqs = []
for i in idx:
    p, s, lab = parse_conv(tbl[int(i)]['conversations'])
    ids, bounds = build(p, s)
    if ids is not None and len(bounds) >= 1 and len(bounds) == len(lab):
        seqs.append((ids, bounds, np.array(lab, np.float32)))
print(f"train seqs={len(seqs)}", flush=True)

params = [p for p in model.parameters() if p.requires_grad] + list(head_last.parameters()) + list(head_9.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)
alllab = np.concatenate([s[2] for s in seqs]); posw = torch.tensor((1 - alllab.mean()) / max(alllab.mean(), 1e-3), device='cuda')
lf = nn.BCEWithLogitsLoss(pos_weight=posw)
rng = np.random.default_rng(1)
model.train(); head_last.train(); head_9.train()
losses = []
for step in range(NSTEPS):
    ids, bounds, lab = seqs[rng.integers(len(seqs))]
    lg = logits_of(ids, bounds, True)
    loss = lf(lg, torch.tensor(lab, device='cuda'))
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    if (step + 1) % 200 == 0:
        print(f"step {step+1}/{NSTEPS} loss={np.mean(losses[-200:]):.4f}", flush=True)

model.eval(); head_last.eval(); head_9.eval()
model.save_pretrained('fthead_lora')
torch.save({'head_last': head_last.state_dict(), 'head_9': head_9.state_dict()}, 'fthead_heads.pt')
print("saved fthead_lora + heads", flush=True)

# ---- eval on FULL ProcessBench ----
EVAL = ['pb_gsm8k_full_in', 'pb_math_full_in', 'pb_olympiadbench_full_in', 'pb_omnimath_full_in']


def probs(DATA):
    cases = []
    for f in [x for x in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(x)]:
        d = json.load(open(os.path.join(f, 'sample.json')))
        ids, bounds = build(d['problem'], d['steps'])
        if not bounds:
            continue
        lg = logits_of(ids, bounds, False)
        cases.append((d.get('label', -1) if d.get('label', -1) is None or d.get('label', -1) < len(bounds) else -1,
                      torch.sigmoid(lg).cpu().numpy()))
    return cases


# threshold from a held-out slice of train (localization on val convs)
val = seqs[-300:]
val_cases = [((next((i for i, v in enumerate(lab) if v == 1), -1)), torch.sigmoid(logits_of(ids, bounds, False)).cpu().numpy())
             for ids, bounds, lab in val]


def locf1(cases, t):
    eh = et = ch = ct = 0
    for lab, P in cases:
        pred = next((i for i, p in enumerate(P) if p > t), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


bt, bf = 0.5, -1
for t in np.arange(0.1, 0.95, 0.02):
    f = locf1(val_cases, t)
    if f > bf:
        bf, bt = f, t
print(f"\nfinal loss={np.mean(losses[-200:]):.4f}  val-thr={bt:.2f}", flush=True)
print(f"{'dataset':22s} {'FT-head F1':>10}  (GenPRM paper)")
ref = {'pb_gsm8k_full_in': 52.8, 'pb_math_full_in': 66.6, 'pb_olympiadbench_full_in': 55.1, 'pb_omnimath_full_in': 54.5}
fs = []
for D in EVAL:
    f1 = locf1(probs(D), bt); fs.append(f1)
    print(f"{D:22s} {f1:10.1f}   ({ref[D]})", flush=True)
print(f"{'MEAN':22s} {np.mean(fs):10.1f}   ({np.mean(list(ref.values())):.1f})", flush=True)
