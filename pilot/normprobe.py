"""训练无关的'正常性探针'(HF精确版):冻结生成模型,算每步'给定前提的平均NLL(意外度)'。
异常步(高NLL)= 疑似首错。借鉴传统单类异常检测:正常=可从前提推出(低意外),错误=偏离。
输出阈值无关诊断: 错误案真首错步的 top-1 定位率 / 平均百分位; 再扫阈值给F1上界。
用法: python normprobe.py <data_dir> [limit] [model]
"""
import json, glob, os, sys, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA  = sys.argv[1] if len(sys.argv) > 1 else 'pb_math_120_in'
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'models/DSR1-1.5B'
MAXLEN = 4096

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()


def build(sample):
    steps = [s for s in sample['steps'] if s is not None]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    full_ids = list(tok(sample['problem'] + "\n", add_special_tokens=True).input_ids)
    bounds = []
    for s in steps:
        start = len(full_ids)
        full_ids += tok(s + "\n", add_special_tokens=False).input_ids
        bounds.append((start, len(full_ids)))
    if len(full_ids) > MAXLEN:
        full_ids = full_ids[:MAXLEN]
        bounds = [(a, min(b, MAXLEN)) for a, b in bounds if a < MAXLEN]
    return full_ids, bounds


def step_nlls(full_ids, bounds):
    ids = torch.tensor([full_ids], device='cuda')
    with torch.no_grad():
        logits = model(ids).logits[0]                      # [T, V]
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)  # predict tok j from j-1
        tok_lp = lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)  # nll idx j -> tok_lp[j-1]
    scores = []
    for (a, b) in bounds:
        js = [j for j in range(max(a, 1), b)]
        scores.append(float(-tok_lp[[j - 1 for j in js]].mean()) if js else 0.0)
    return scores


folders = [f for f in sorted(glob.glob(os.path.join(DATA, '*'))) if os.path.isdir(f)][:LIMIT]
records = []
for f in folders:
    d = json.load(open(os.path.join(f, 'sample.json')))
    fids, bounds = build(d)
    if len(bounds) < 1:
        continue
    records.append((os.path.basename(f), d.get('label', -1), len(bounds), step_nlls(fids, bounds)))


def zscores(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs), 1)) or 1e-9
    return [(x - m) / sd for x in xs]


top1 = nerr = 0; pct_sum = chance = 0.0
for name, label, nst, scores in records:
    if label is None or label < 0 or nst < 2:
        continue
    nerr += 1
    amax = max(range(nst), key=lambda i: scores[i])
    top1 += (amax == label)
    pct_sum += sum(1 for s in scores if s <= scores[label]) / nst
    chance += 1.0 / nst
print(f"[{DATA}] model={MODEL}  n={len(records)}  error_cases={nerr}")
if nerr:
    print(f"  top1-localization = {top1/nerr*100:.1f}%   (chance ~ {chance/nerr*100:.1f}%)")
    print(f"  mean percentile of true-error step = {pct_sum/nerr*100:.1f}%   (chance 50%)")

best = (0.0, None)
for tau10 in range(-5, 31):
    tau = tau10 / 10
    eh = et = ch = ct = 0
    for name, label, nst, scores in records:
        z = zscores(scores) if nst > 1 else [0.0]
        pred = next((i for i, zz in enumerate(z) if zz > tau), -1)
        if label == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == label)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    f1 = 2 * ae * ac / max(ae + ac, 1e-9)
    if f1 > best[0]:
        best = (f1, (tau, ae, ac, et, ct))
if best[1] is None:
    print("  oracle-threshold F1: n/a (no correct cases in slice)")
else:
    f1, (tau, ae, ac, et, ct) = best
    print(f"  oracle-threshold F1 upper-bound = {f1*100:.1f}  (tau={tau:.1f} accE={ae*100:.1f} accC={ac*100:.1f} nerr={et} ncor={ct})")
