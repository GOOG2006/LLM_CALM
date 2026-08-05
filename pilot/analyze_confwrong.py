"""自信错(自洽但错)vs 真值的差异结构分析。不发 API 调用。"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]

conf_wrong = [r for r in rows if r["correct"] == 0 and r["base_disagree"] == 0.0]
all_wrong = [r for r in rows if r["correct"] == 0]
print(f"总 N={len(rows)}  总错={len(all_wrong)}  自信错(自洽且错)={len(conf_wrong)}")
print(f"自信错 cross_disagree 均值={sum(r['cross_disagree'] for r in conf_wrong)/max(len(conf_wrong),1):.2f} "
      f"(=0 表示 checker 也同样错)")
print("=" * 70)

def classify(b, g):
    if b is None or g is None:
        return "no-answer"
    if g != 0:
        r = b / g
        for k, name in [(2, "×2"), (0.5, "÷2"), (10, "×10"), (0.1, "÷10"),
                        (3, "×3"), (1/3, "÷3"), (100, "×100")]:
            if abs(r - k) < 1e-6:
                return f"整倍数 {name}"
    d = b - g
    if abs(d) <= 3 and d == int(d):
        return f"小加性偏移 {d:+.0f}"
    if g != 0 and abs(b) > 0:
        import math
        if abs(math.log10(abs(b)) - math.log10(abs(g))) >= 1:
            return "量级错(≥10×)"
    return "无明显结构"

from collections import Counter
cats = Counter()
print(f"{'id':>5} {'gold':>10} {'base_ans':>10} {'ratio':>8}  {'diff':>10}  类别")
for r in sorted(conf_wrong, key=lambda x: x["id"]):
    b, g = r["base_ans"], r["gold"]
    ratio = (b / g) if (b is not None and g not in (None, 0)) else float("nan")
    diff = (b - g) if (b is not None and g is not None) else float("nan")
    c = classify(b, g)
    cats[c] += 1
    print(f"{r['id']:>5} {g:>10.2f} {b if b is not None else float('nan'):>10.2f} "
          f"{ratio:>8.3f}  {diff:>10.2f}  {c}")
print("-" * 70)
print("类别分布:", dict(cats))
