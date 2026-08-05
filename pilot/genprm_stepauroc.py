"""GenPRM full-scores jsonl 的 step 级 error-AUROC(与 compare_sameset 同口径)。中间进度可用。
用法: python genprm_stepauroc.py out_olymp40.jsonl"""
import sys, json

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]; neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return float("nan")
    w = t = 0
    for p in pos:
        for n in neg:
            if p > n: w += 1
            elif p == n: t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
el, lab = [], []
ncase_err = ncase_cor = 0
for r in rows:
    L = int(r["label"]); sc = r["scores"]
    if L == -1: ncase_cor += 1
    else: ncase_err += 1
    for t in range(len(sc)):
        if L >= 0 and t > L: continue
        el.append(1 - sc[t]); lab.append(int(L >= 0 and t == L))
print(f"cases done={len(rows)} (err{ncase_err}/cor{ncase_cor})  steps={len(lab)} pos={sum(lab)}")
print(f"GenPRM step error-AUROC = {auroc(el, lab):.3f}")
