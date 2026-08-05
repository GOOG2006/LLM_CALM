"""从已落盘的 results jsonl 直接算判决(不再发任何 API 调用)。"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")

def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])
    rank, i = {}, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            rank[ranked[k][1]] = avg
        i = j
    rpos = sum(rank[idx] for idx, y in enumerate(labels) if y == 1)
    u = rpos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
labels = [1 - r["ref_correct"] for r in rows]  # 正类=答错
n_wrong = sum(labels); n_right = len(labels) - n_wrong
auc_red = auroc([r["off_rate_rederive"] for r in rows], labels)
auc_fresh = auroc([r["disagree_fresh"] for r in rows], labels)
lift = auc_red - auc_fresh

# 错误桶内,off_rate 的分布(看'自洽但错'有多普遍:错答却 off_rate=0 = 稳定错盆地)
wrong_off = [r["off_rate_rederive"] for r in rows if r["ref_correct"] == 0]
right_off = [r["off_rate_rederive"] for r in rows if r["ref_correct"] == 1]
def mean(x): return sum(x)/len(x) if x else float("nan")
wrong_stable = sum(1 for o in wrong_off if o == 0.0)  # 错答但重推导零脱落 = 自洽但错

print(f"有效样本 N={len(rows)}   答错={n_wrong}   答对={n_right}   (错误桶需≥15)")
print(f"AUROC[中点重推导 off_rate] = {auc_red:.3f}   ← 主指标")
print(f"AUROC[全新生成 disagree]   = {auc_fresh:.3f}   ← 对照(self-consistency)")
print(f"闸①增益 (重推导 - 对照)    = {lift:+.3f}   ← 真判决量 (≥0.03 才有独立价值)")
print(f"错答 off_rate 均值={mean(wrong_off):.3f}  对答 off_rate 均值={mean(right_off):.3f}")
print(f"'自洽但错'(错答却零脱落): {wrong_stable}/{n_wrong} = {wrong_stable/max(n_wrong,1):.0%}")
print("-"*52)
verdict = ("过闸→进 Gate2" if (auc_red >= 0.65 and lift >= 0.03) else
           "判死(self-consistency 换皮)" if (auc_red >= 0.65 and lift < 0.03) else
           "判死(无判别力)" if auc_red < 0.55 else "灰区→换 MATH")
print(f"裁决: {verdict}")
