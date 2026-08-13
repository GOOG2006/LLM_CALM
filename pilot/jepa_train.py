"""JEPA式自监督预测编码 LoRA 微调(冻结目标变体, 不坍缩)。
目标: online(base+LoRA) 的'前缀表示' 预测 target(冻结base) 的'下一步表示'(隐空间, stop-grad)。
学'从前提可推出下一步' -> 推理时偏离=错误。训练用步序列(GenPRM-Data, 无需标签)。
用法: python jepa_train.py [n_seq] [steps]
"""
import re, sys, pickle, torch, numpy as np
import torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = 'models/DSR1-1.5B'
LAYER, MAXLEN = -9, 2048
NSEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                  lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, lora)
model.train()
H = model.config.hidden_size
predictor = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H)).cuda().float()


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']
    if not users:
        return None, None
    u0 = re.sub(r'^\s*Question:\s*', '', users[0])
    problem, step0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    return problem, [step0] + users[1:]


def build(problem, steps):
    steps = [s for s in steps if isinstance(s, str) and s]
    if len(steps) < 2:
        return None, None
    ids = list(tok(problem + "\n", add_special_tokens=True).input_ids)
    bounds = []
    for s in steps:
        a = len(ids); ids += tok(s + "\n", add_special_tokens=False).input_ids
        bounds.append((a, len(ids)))
    if len(ids) > MAXLEN:
        return None, None
    return ids, bounds


tbl = pq.read_table('GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
idx = np.random.default_rng(0).permutation(len(tbl))[:NSEQ]
seqs = []
for i in idx:
    p, s = parse_conv(tbl[int(i)]['conversations'])
    if p is None:
        continue
    ids, bounds = build(p, s)
    if ids is not None and len(bounds) >= 2:
        seqs.append((ids, bounds))
print(f"sequences={len(seqs)}", flush=True)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(predictor.parameters()), lr=1e-4)
sl1 = nn.SmoothL1Loss()
rng = np.random.default_rng(1)


def hiddens(ids, use_lora):
    t = torch.tensor([ids], device='cuda')
    if use_lora:
        return model(t, output_hidden_states=True).hidden_states[LAYER][0]      # grad on
    with torch.no_grad(), model.disable_adapter():
        return model(t, output_hidden_states=True).hidden_states[LAYER][0]       # frozen target


losses = []
for step in range(NSTEPS):
    ids, bounds = seqs[rng.integers(len(seqs))]
    hs_on = hiddens(ids, True)                        # online (LoRA), grad
    with torch.no_grad():
        hs_tg = hiddens(ids, False).float()          # target (frozen), stop-grad
    C, Z = [], []
    for (a, b) in bounds[1:]:                         # predict each step from its prefix
        C.append(hs_on[a - 1])                        # prefix last token
        Z.append(hs_tg[a:b].mean(0))                  # target: mean of this step
    C = torch.stack(C).float(); Z = torch.stack(Z)
    pred = predictor(C)
    loss = sl1(pred, Z)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if (step + 1) % 100 == 0:
        print(f"step {step+1}/{NSTEPS} loss={np.mean(losses[-100:]):.4f}", flush=True)

model.save_pretrained('jepa_lora')
torch.save(predictor.state_dict(), 'jepa_predictor.pt')
print("saved jepa_lora + predictor. final loss=%.4f" % np.mean(losses[-100:]), flush=True)
