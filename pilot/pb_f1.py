"""ProcessBench 标准 F1(case 级找首错步)对比:GenPRM vs ours。
F1 = harmonic mean(错误案例正确定位首错步的acc, 正确案例判全对的acc)。
ours 连续分 s_mean 需阈值 tau(s<tau 判该步错);扫 tau 诚实呈现敏感性。
用法: python pb_f1.py results_think_sc_v2.jsonl out_math60.jsonl
"""
import sys, json
from collections import defaultdict

ours = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
gp = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()}

ocase = defaultdict(list)
for r in ours:
    ocase[r["cid"]].append((r["t"], r["s_mean"]))
for c in ocase:
    ocase[c].sort()

cases = [cid for cid in ocase if cid in gp]
errc = [cid for cid in cases if int(gp[cid]["label"]) >= 0]
corc = [cid for cid in cases if int(gp[cid]["label"]) == -1]


def f1(ea, ca):
    return 2 * ea * ca / (ea + ca) if (ea + ca) > 0 else 0.0


def ours_pred(c, tau):
    return next((t for t, s in ocase[c] if s < tau), -1)


def gp_pred(c, thr):
    sc = gp[c]["scores"]
    for t, v in enumerate(sc):
        if v < thr:
            return t
    return -1


print(f"cases: err={len(errc)} cor={len(corc)}\n")

# GenPRM 标准 thr=0.5(也扫几个)
print("=== GenPRM-1.5B ===")
for thr in [0.5, 0.4, 0.6]:
    ea = sum(gp_pred(c, thr) == int(gp[c]["label"]) for c in errc) / max(len(errc), 1)
    ca = sum(gp_pred(c, thr) == -1 for c in corc) / max(len(corc), 1)
    print(f"  thr={thr}: err-acc={ea:.3f} cor-acc={ca:.3f} F1={f1(ea,ca)*100:.1f}")

print("\n=== ours SC think-measure (tau 扫描) ===")
best = (0, None)
for tau in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
    ea = sum(ours_pred(c, tau) == int(gp[c]["label"]) for c in errc) / max(len(errc), 1)
    ca = sum(ours_pred(c, tau) == -1 for c in corc) / max(len(corc), 1)
    F = f1(ea, ca)
    if F > best[0]:
        best = (F, tau)
    print(f"  tau={tau:>4}: err-acc={ea:.3f} cor-acc={ca:.3f} F1={F*100:.1f}")
print(f"\n  ours 最佳 F1={best[0]*100:.1f} @ tau={best[1]}")
