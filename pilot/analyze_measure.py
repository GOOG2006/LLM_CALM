"""
后处理 results_measure_math10.jsonl:测 案内归一化 + 组合算子 的 error-AUROC。
假说:错误步 = 它自己那条链里的离群点 → 案内 z-score 去题目难度混淆,应比 raw 更强。
用法: python analyze_measure.py results_measure_math10.jsonl [out_math10.jsonl]
"""
import sys, json
from statistics import mean, pstdev
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
KEYS = [k for k in ["mean_logp", "min_logp", "mean_ent", "max_ent", "ent_p90", "ent_var",
                    "mean_margin", "max_margin"] if k in rows[0]]
BIG_IS_ERR = {"mean_ent", "max_ent", "ent_p90", "ent_var", "mean_margin", "max_margin"}

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return float("nan")
    w = t = 0
    for p in pos:
        for n in neg:
            if p > n: w += 1
            elif p == n: t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))

def best(scores, labels):
    a = auroc(scores, labels); b = auroc([-x for x in scores], labels)
    return (max(a, b), "raw大=错" if a >= b else "反号小=错")

labels = [r["is_err"] for r in rows]

# 案内 z-score
bycid = defaultdict(list)
for i, r in enumerate(rows):
    bycid[r["cid"]].append(i)
def zcol(key):
    z = [0.0] * len(rows)
    for cid, idxs in bycid.items():
        vals = [rows[i][key] for i in idxs]
        if len(vals) < 2:
            continue
        mu = mean(vals); sd = pstdev(vals) or 1e-9
        for i in idxs:
            z[i] = (rows[i][key] - mu) / sd
    return z

print(f"N_steps={len(rows)} 错步={sum(labels)}\n")
print("=== raw vs 案内z-score ===")
zcache = {}
for k in KEYS:
    au_raw, d = best([r[k] for r in rows], labels)
    zc = zcol(k); zcache[k] = zc
    au_z, dz = best(zc, labels)
    print(f"  {k:<12} raw={au_raw:.3f}[{d}]   z={au_z:.3f}[{dz}]")

# 全局 z-score(跨所有步),方向对齐成“大=错”
def gz(key):
    vals = [r[key] for r in rows]
    mu = mean(vals); sd = pstdev(vals) or 1e-9
    z = [(v - mu) / sd for v in vals]
    return z if key in BIG_IS_ERR else [-x for x in z]

def signed_zc(k):  # 案内 z,对齐大=错
    return zcache[k] if k in BIG_IS_ERR else [-x for x in zcache[k]]

combos = {
    "ent+neglogp": ["mean_ent", "mean_logp"],
    "ent+margin": ["mean_ent", "mean_margin"],
    "entp90+neglogp": ["ent_p90", "mean_logp"],
    "ent+neglogp+margin": ["mean_ent", "mean_logp", "mean_margin"],
    "all": KEYS,
}
print("\n=== 组合(全局z, 方向对齐相加) ===")
for name, ks in combos.items():
    ks = [k for k in ks if k in rows[0]]
    comb = [sum(gz(k)[i] for k in ks) for i in range(len(rows))]
    print(f"  {name:<22} AUROC={best(comb, labels)[0]:.3f}")
print("\n=== 组合(案内z, 方向对齐相加) ===")
for name, ks in combos.items():
    ks = [k for k in ks if k in rows[0]]
    comb = [sum(signed_zc(k)[i] for k in ks) for i in range(len(rows))]
    print(f"  {name:<22} AUROC={best(comb, labels)[0]:.3f}")

# GenPRM 参考
if len(sys.argv) > 2:
    gp = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()}
    gel, gl = [], []
    for r in rows:
        g = gp.get(r["cid"])
        if g and r["t"] < len(g["scores"]):
            gel.append(1 - g["scores"][r["t"]]); gl.append(r["is_err"])
    print(f"\n  GenPRM (1-score) AUROC={auroc(gel, gl):.3f}")
