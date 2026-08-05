"""
同底物公平对比:把 ours(SC think-measure)限制到 GenPRM 已评测的同一批步(out_math10 的 49 步),
在完全相同的 (cid,t) 上比 AUROC。避免跨集/功效不一致。
用法: python compare_sameset.py results_think_sc30.jsonl out_math10.jsonl
"""
import sys, json

ours = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
gp = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()}


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


# 限制到 GenPRM 已评测的同一批步
rows = []
for r in ours:
    g = gp.get(r["cid"])
    if g and r["t"] < len(g["scores"]):
        rows.append((r, g))
labs = [r["is_err"] for r, _ in rows]

# 覆盖率:GenPRM 全部非污染步 vs ours 覆盖到的步
gp_total = 0
for g in gp.values():
    L = int(g["label"]); ns = len(g["scores"])
    gp_total += (L + 1) if L >= 0 else ns
ours_cids = {r["cid"] for r in ours}
gp_cids = set(gp.keys())
print(f"同底物重叠步={len(rows)}  错步={sum(labs)}")
print(f"覆盖率: ours 覆盖 {len(rows)}/{gp_total} = {100*len(rows)/max(gp_total,1):.1f}% 的 GenPRM 非污染步  "
      f"(案例: ours {len(ours_cids & gp_cids)}/{len(gp_cids)})\n")

# GenPRM
gp_el = [1 - g["scores"][r["t"]] for r, g in rows]
print(f"  GenPRM-1.5B (1-score)          AUROC={auroc(gp_el, labs):.3f}")

# ours: samples 曲线 + s_mean
S = rows[0][0].get("svals")
nmax = len(S) if S else 1
for j in range(1, nmax + 1):
    el = [-(sum(r["svals"][:j]) / j) for r, _ in rows]
    print(f"  ours SC think-measure s={j}     AUROC={auroc(el, labs):.3f}")
