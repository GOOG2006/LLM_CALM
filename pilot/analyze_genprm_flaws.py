"""
量化 GenPRM 缺陷 W1/W2/W5/W7 + ProcessBench F1 基线。
读 genprm_run.py --full 的输出(每案全步 scores)。
用法: python analyze_genprm_flaws.py out_math_full.jsonl [out_olympiad_full.jsonl ...] --thr 0.5
"""
import sys, json, argparse
from statistics import mean, pstdev


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pred_from_scores(scores, thr):
    """首个 score<thr 的 1-indexed 步;都>=thr → -1(判全对)。返回 0-indexed 与 gold 对齐。"""
    for i, s in enumerate(scores):
        if s < thr:
            return i
    return -1


def prf(rows, thr):
    err = [r for r in rows if int(r["label"]) != -1]
    cor = [r for r in rows if int(r["label"]) == -1]
    ea = mean([pred_from_scores(r["scores"], thr) == int(r["label"]) for r in err]) if err else float("nan")
    ca = mean([pred_from_scores(r["scores"], thr) == -1 for r in cor]) if cor else float("nan")
    f1 = 2 * ea * ca / (ea + ca) if (ea + ca) else float("nan")
    return ea, ca, f1, len(err), len(cor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    rows = load(a.paths)
    print(f"cases={len(rows)}  files={a.paths}\n")

    # ===== ProcessBench F1(baseline 数字) =====
    ea, ca, f1, ne, nc = prf(rows, a.thr)
    print(f"===== ProcessBench-style (thr={a.thr}) =====")
    print(f"error-acc={ea:.3f} correct-acc={ca:.3f}  F1={f1*100:.1f}  (err={ne}/cor={nc})\n")

    # 阈值敏感度(与 W1/W5 相关:分数不校准 → F1 随 thr 抖)
    print("阈值敏感度 F1:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        _, _, f, _, _ = prf(rows, t)
        print(f"  thr={t}: F1={f*100:.1f}")
    print()

    # ===== W1 过自信/饱和 =====
    allsc = [s for r in rows for s in r["scores"]]
    sat = mean([s >= 0.9 for s in allsc])
    half = mean([abs(s - 0.5) < 1e-6 for s in allsc])
    print("===== W1 饱和 / W5 中性回退 =====")
    print(f"全部步分数 N={len(allsc)}  均值={mean(allsc):.3f}  在[0.9,1.0]占比={sat*100:.1f}%  ==0.5占比(W5)={half*100:.1f}%")
    # 完全漏检:final_answer_correct=false(确有错) 却所有步 >=thr → 判全对
    wrongcases = [r for r in rows if str(r.get("final_answer_correct")).lower() == "false"]
    miss = [r for r in wrongcases if all(s >= a.thr for s in r["scores"])]
    print(f"确有错案例(final_answer_correct=false)={len(wrongcases)}  其中所有步判过关(完全漏检)={len(miss)} "
          f"({100*len(miss)/max(len(wrongcases),1):.1f}%)\n")

    # ===== W2 早承诺(仅错误案例) =====
    err = [r for r in rows if int(r["label"]) != -1]
    bias = []  # 预测首错步(0idx) - 真首错步
    for r in err:
        p = pred_from_scores(r["scores"], a.thr)
        if p != -1:
            bias.append(p - int(r["label"]))
    early = mean([b < 0 for b in bias]) if bias else float("nan")
    print("===== W2 早承诺 =====")
    print(f"有预测的错误案例={len(bias)}  预测早于真首错(bias<0)占比={early*100:.1f}%  bias均值={mean(bias):.2f}" if bias else "无")
    # 漏检率 vs 链长
    def bin_miss(lo, hi):
        sub = [r for r in err if lo <= r["n_steps"] <= hi]
        if not sub:
            return None
        m = mean([pred_from_scores(r["scores"], a.thr) != int(r["label"]) for r in sub])
        return len(sub), m
    print("漏检率(未命中真首错) vs 链长:")
    for lo, hi, name in [(1, 6, "短≤6"), (7, 12, "中7-12"), (13, 99, "长≥13")]:
        b = bin_miss(lo, hi)
        if b:
            print(f"  {name}: n={b[0]} miss={b[1]:.3f}")
    print()

    # ===== W7 算力-精度 =====
    if all("sec" in r for r in rows):
        print("===== W7 算力 vs 链长 =====")
        for lo, hi, name in [(1, 6, "短≤6"), (7, 12, "中7-12"), (13, 99, "长≥13")]:
            sub = [r for r in rows if lo <= r["n_steps"] <= hi]
            if sub:
                print(f"  {name}: n={len(sub)} 平均 {mean([r['sec'] for r in sub]):.1f}s  "
                      f"平均步数 {mean([r['n_steps'] for r in sub]):.1f}")


if __name__ == "__main__":
    main()
