"""
融合 think-measure 判决 logit(results_think_vllm.jsonl)与静态测量(results_measure_math60.jsonl)。
join on (cid,t),测单信号与全局-z 组合的 error-AUROC。对比 GenPRM-1.5B(out_math10.jsonl)。
用法: python combine_signals.py results_think_vllm.jsonl results_measure_math60.jsonl out_math10.jsonl
"""
import sys, json, math
from statistics import mean, pstdev

think = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
meas = {(json.loads(l)["cid"], json.loads(l)["t"]): json.loads(l)
        for l in open(sys.argv[2], encoding="utf-8") if l.strip()}


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    w = t = 0
    for p in pos:
        for n in neg:
            if p > n: w += 1
            elif p == n: t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


# join
rows = []
for r in think:
    m = meas.get((r["cid"], r["t"]))
    if not m:
        continue
    s = r["s"]
    rows.append(dict(cid=r["cid"], t=r["t"], is_err=r["is_err"],
                     think512=s.get("512", s.get(512)), think256=s.get("256", s.get(256)),
                     mean_ent=m["mean_ent"], min_logp=m["min_logp"], mean_logp=m["mean_logp"]))
labels = [r["is_err"] for r in rows]
print(f"join steps={len(rows)} pos={sum(labels)}\n")

# error-likelihood 方向:think→ -s(Yes低=错); ent→ +(大=错); logp→ -(小=错)
def el(key):
    if key.startswith("think"):
        return [-r[key] for r in rows]
    if "ent" in key:
        return [r[key] for r in rows]
    return [-r[key] for r in rows]  # logp

def gz(vals):
    mu = mean(vals); sd = pstdev(vals) or 1e-9
    return [(v - mu) / sd for v in vals]

print("=== 单信号 ===")
for k in ["think512", "think256", "mean_ent", "min_logp", "mean_logp"]:
    print(f"  {k:<10} AUROC={auroc(el(k), labels):.3f}")

print("\n=== 全局-z 组合(error方向对齐相加) ===")
combos = {
    "think512+ent": ["think512", "mean_ent"],
    "think512+minlogp": ["think512", "min_logp"],
    "think512+ent+minlogp": ["think512", "mean_ent", "min_logp"],
    "think512+think256+ent": ["think512", "think256", "mean_ent"],
}
for name, ks in combos.items():
    comb = [sum(gz(el(k))[i] for k in ks) for i in range(len(rows))]
    print(f"  {name:<24} AUROC={auroc(comb, labels):.3f}")

# GenPRM 参考(overlap)
if len(sys.argv) > 3:
    gp = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[3], encoding="utf-8") if l.strip()}
    gel, gl = [], []
    for r in rows:
        g = gp.get(r["cid"])
        if g and r["t"] < len(g["scores"]):
            gel.append(1 - g["scores"][r["t"]]); gl.append(r["is_err"])
    if gl:
        print(f"\n  [重叠{len(gl)}步] GenPRM-1.5B AUROC={auroc(gel, gl):.3f}")
