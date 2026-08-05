"""
头对头:GenPRM(生成式判决)vs Detailed-Balance(测量式 σ),同一批 math 步。
- GenPRM error-likelihood = 1 - score
- Ours error-likelihood     = sigma(方向自适应)
指标: step 级 error-AUROC(阈值无关,主指标) + 错误案例首错步定位命中(argmax)。
用法: python compare_two.py out_math10.jsonl results_detbal_math10.jsonl
"""
import sys, json


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


def best_auroc(scores, labels):
    a = auroc(scores, labels)
    b = auroc([-x for x in scores], labels)
    return (a, "raw(大=错)") if a >= b else (b, "反号(小=错)")


def main():
    gp_path, ours_path = sys.argv[1], sys.argv[2]
    gp = {}
    for line in open(gp_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); gp[r["id"]] = r
    ours = [json.loads(l) for l in open(ours_path, encoding="utf-8") if l.strip()]

    # 对齐:每个 ours 行(cid,t) 取 GenPRM 同步分数
    is_err, gp_el, our_el = [], [], []
    matched = set()
    for r in ours:
        cid, t = r["cid"], r["t"]
        g = gp.get(cid)
        if not g or t >= len(g["scores"]):
            continue
        matched.add(cid)
        is_err.append(r["is_err"])
        gp_el.append(1.0 - g["scores"][t])   # GenPRM: 分数越低越像错
        our_el.append(r["sigma"])            # Ours: σ
    npos = sum(is_err)
    print(f"对齐案例={len(matched)}  评分步数={len(is_err)}  其中错误步={npos}\n")

    a_gp = auroc(gp_el, is_err)
    a_our, dir_our = best_auroc(our_el, is_err)
    print("===== step 级 error-AUROC(主指标,阈值无关)=====")
    print(f"  GenPRM (1-score)      AUROC = {a_gp:.3f}")
    print(f"  Ours   (σ, {dir_our}) AUROC = {a_our:.3f}")
    print(f"  Δ(ours - genprm) = {a_our - a_gp:+.3f}\n")

    # 首错步定位命中(仅错误案例):谁把"最像错的步"指到真首错步
    err_cases = {}
    for r in ours:
        g = gp.get(r["cid"])
        if not g or int(g["label"]) == -1:
            continue
        err_cases.setdefault(r["cid"], []).append(r)
    gp_hit = our_hit = ntot = 0
    for cid, rows in err_cases.items():
        g = gp[cid]; lab = int(g["label"])
        rows = [x for x in rows if x["t"] < len(g["scores"])]
        if not rows:
            continue
        ntot += 1
        # GenPRM: 分数最低的步
        gp_pick = min(rows, key=lambda x: g["scores"][x["t"]])["t"]
        # Ours: σ 最大(或最小,取全局方向 dir_our)的步
        our_pick = (max(rows, key=lambda x: x["sigma"]) if dir_our.startswith("raw")
                    else min(rows, key=lambda x: x["sigma"]))["t"]
        gp_hit += (gp_pick == lab); our_hit += (our_pick == lab)
    if ntot:
        print("===== 首错步定位命中率(错误案例, argmax)=====")
        print(f"  GenPRM: {gp_hit}/{ntot} = {gp_hit/ntot:.2f}")
        print(f"  Ours  : {our_hit}/{ntot} = {our_hit/ntot:.2f}")

    print("\n[注] 10 案为 smoke,功效低(错误步~个位数),仅看方向;有信号再扩样。")


if __name__ == "__main__":
    main()
