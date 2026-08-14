"""7B 残差判词头 + LoRA: logit = Linear(h_last) + MLP(h_9^步末)。基座=DSR1-7B(GenPRM-7B 的基座)。
在 GenPRM-Data 步级 Yes/No 上训。eval 在 pb_*_7b_out 的同一批 120 案(和 GenPRM-7B P_B 逐案可比)。
bf16 + 梯度检查点省显存。用法: python finetune_head_7b.py [n_conv] [nsteps]
"""
import json, glob, os, sys, re, torch, numpy as np
import torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

MODEL = 'models/DSR1-7B'; L9, MAXLEN = -9, 1792
NCONV = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12000

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
base.gradient_checkpointing_enable(); base.config.use_cache = False
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
model.enable_input_require_grads()
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
        return head_last(hl[idxs]).squeeze(-1) + head_9(h9[idxs]).squeeze(-1)


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

ref = {'pb_gsm8k_7b_out': 83.4, 'pb_math_7b_out': 80.0, 'pb_olympiadbench_7b_out': 72.3, 'pb_omnimath_7b_out': 71.5}


def probs_from_out(OUT):
    cases = []
    for fo in sorted(glob.glob(os.path.join(OUT, '*_analyze'))):
        rp = os.path.join(fo, 'result_1.json')
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        ids, bounds = build(r['problem'], r['steps'])
        if not bounds:
            continue
        PA = torch.sigmoid(logits_of(ids, bounds, False)).cpu().numpy()   # fthead P(error)
        PB = np.array(r['value'], dtype=float)                            # GenPRM P(correct)
        lab = r['label'] if (r['label'] == -1 or r['label'] < len(bounds)) else -1
        cases.append((lab, PA, PB))
    return cases


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
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100


def fuse_report(cases, tA):
    # evidence_A = logit(PA)-logit(tA); evidence_B = logit(1-PB); pred = first step evidence>bias
    fA = f1_pred(cases, lambda PA, PB: next((i for i in range(len(PA)) if PA[i] > tA), -1))
    fB = f1_pred(cases, lambda PA, PB: next((i for i in range(len(PB)) if PB[i] < 0.5), -1))
    def fused(a, bias):
        return lambda PA, PB: next((i for i in range(min(len(PA), len(PB)))
                                    if a * (_lz(PA[i]) - _lz(tA)) + (1 - a) * _lz(1 - PB[i]) > bias), -1)
    f5 = f1_pred(cases, fused(0.5, 0.0))
    best = 0.0
    for a in np.arange(0, 1.01, 0.1):
        for bias in np.arange(-2, 2.01, 0.25):
            best = max(best, f1_pred(cases, fused(a, bias)))
    return fA, fB, f5, best


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


def evaluate(tag):
    model.eval(); head_last.eval(); head_9.eval()
    # threshold from a LARGER held-out slice (stable), + a fixed-0.5 read for a clean trajectory
    val = seqs[-800:]
    val_cases = [((next((i for i, v in enumerate(lab) if v == 1), -1)), torch.sigmoid(logits_of(ids, bounds, False)).cpu().numpy())
                 for ids, bounds, lab in val]
    bt, bf = 0.5, -1
    for t in np.arange(0.30, 0.85, 0.02):   # clamp away from extreme thresholds
        f = locf1(val_cases, t)
        if f > bf:
            bf, bt = f, t
    outs = ['pb_gsm8k_7b_out', 'pb_math_7b_out', 'pb_olympiadbench_7b_out', 'pb_omnimath_7b_out']
    print(f"\n===== EVAL @ {tag}  val-thr={bt:.2f} =====", flush=True)
    print(f"{'dataset':22s} {'fthead':>7} {'GenPRM':>7} {'fuse.5':>7} {'fuseOR':>7}", flush=True)
    A, B, F5, OR = [], [], [], []
    for D in outs:
        cs = probs_from_out(D)
        fA, fB, f5, best = fuse_report(cs, bt)
        A.append(fA); B.append(fB); F5.append(f5); OR.append(best)
        print(f"{D:22s} {fA:7.1f} {fB:7.1f} {f5:7.1f} {best:7.1f}", flush=True)
    print(f"{'MEAN':22s} {np.mean(A):7.1f} {np.mean(B):7.1f} {np.mean(F5):7.1f} {np.mean(OR):7.1f}", flush=True)
    model.train(); head_last.train(); head_9.train()


params = [p for p in model.parameters() if p.requires_grad] + list(head_last.parameters()) + list(head_9.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)
alllab = np.concatenate([s[2] for s in seqs]); posw = torch.tensor((1 - alllab.mean()) / max(alllab.mean(), 1e-3), device='cuda')
lf = nn.BCEWithLogitsLoss(pos_weight=posw)
rng = np.random.default_rng(1)
EVAL_EVERY = 500
CK = 'ck7b.pt'


def save_ck(step):
    torch.save({'lora': get_peft_model_state_dict(model), 'head_last': head_last.state_dict(),
                'head_9': head_9.state_dict(), 'opt': opt.state_dict(),
                'rng': rng.bit_generator.state, 'step': step}, CK + '.tmp')
    os.replace(CK + '.tmp', CK)   # atomic: never leave a half-written ckpt


start_step = 0
if os.path.exists(CK):
    ck = torch.load(CK, map_location='cuda')
    set_peft_model_state_dict(model, ck['lora'])
    head_last.load_state_dict(ck['head_last']); head_9.load_state_dict(ck['head_9'])
    opt.load_state_dict(ck['opt']); rng.bit_generator.state = ck['rng']; start_step = ck['step']
    print(f"RESUMED from step {start_step}", flush=True)

model.train(); head_last.train(); head_9.train()
import time
losses = []; t0 = time.time()
for step in range(start_step, NSTEPS):
    ids, bounds, lab = seqs[rng.integers(len(seqs))]
    lg = logits_of(ids, bounds, True)
    loss = lf(lg, torch.tensor(lab, device='cuda'))
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    if (step + 1) % 200 == 0:
        el = time.time() - t0; done = step + 1 - start_step
        eta = el / done * (NSTEPS - step - 1) / 3600
        print(f"step {step+1}/{NSTEPS} loss={np.mean(losses[-200:]):.4f} {el/done:.2f}s/it eta={eta:.1f}h", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        save_ck(step + 1)
        if (step + 1) < NSTEPS:
            evaluate(f"step{step+1}")
            t0 = time.time(); start_step = step + 1   # exclude eval time from s/it

model.eval(); head_last.eval(); head_9.eval()
model.save_pretrained('fthead7b_lora')
torch.save({'head_last': head_last.state_dict(), 'head_9': head_9.state_dict()}, 'fthead7b_heads.pt')
print(f"\nsaved fthead7b_lora + heads. final loss={np.mean(losses[-200:]):.4f}", flush=True)
evaluate("FINAL")
