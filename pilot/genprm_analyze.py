"""分析 GenPRM 预测:recall/FP/定位错 + 导出漏检残余供裁决。远程运行。
用法: python genprm_analyze.py --config gsm8k --res gsm8k_maj1.jsonl --dump residuals_gsm8k.json"""
import json, argparse
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="gsm8k")
ap.add_argument("--res", required=True)
ap.add_argument("--dump", default="residuals.json")
a = ap.parse_args()

DATADIR = "/root/autodl-tmp/genprm_work/ProcessBench"
gold = {d["id"]: d for d in json.load(open(f"{DATADIR}/{a.config}.json"))}
rows = [json.loads(l) for l in open(a.res) if l.strip()]

err = [r for r in rows if r["label"] != -1]
cor = [r for r in rows if r["label"] == -1]

# hit: error样本 pred_1idx-1==label(ProcessBench 0-indexed首错);correct样本 pred==-1
def hit_err(r): return r["pred_1idx"] - 1 == r["label"]
def hit_cor(r): return r["pred_1idx"] == -1

recall = sum(hit_err(r) for r in err) / max(len(err), 1)
fp = sum(1 for r in cor if not hit_cor(r)) / max(len(cor), 1)
acc_cor = 1 - fp
f1 = 2 * recall * acc_cor / max(recall + acc_cor, 1e-9)  # ProcessBench 式 F1(err-acc, cor-acc 调和)

print(f"config={a.config}  错误N={len(err)}  全对N={len(cor)}")
print(f"error-recall(定位对首错) = {recall:.3f}")
print(f"correct-acc(全对判全对)  = {acc_cor:.3f}   (FP={fp:.3f})")
print(f"ProcessBench-F1          = {f1:.3f}")

# 残余拆分:漏检(pred=-1判全对却有错) vs 定位错(flag了但不是首错步)
missed = [r for r in err if r["pred_1idx"] == -1]
misloc = [r for r in err if r["pred_1idx"] != -1 and not hit_err(r)]
print(f"残余: 完全漏检(判全对)={len(missed)}  定位错(flag错步)={len(misloc)}")

# 导出残余供人工裁决
dump = []
for r in (missed + misloc):
    g = gold[r["id"]]
    gi = g["label"]  # 0-indexed 首错步
    dump.append({
        "id": r["id"], "kind": "missed" if r["pred_1idx"] == -1 else "misloc",
        "gold_error_step_0idx": gi, "pred_1idx": r["pred_1idx"], "n_steps": r["n_steps"],
        "scores": r["scores"], "problem": g["problem"],
        "gold_error_step_text": g["steps"][gi] if 0 <= gi < len(g["steps"]) else "",
        "pred_step_text": g["steps"][r["pred_1idx"] - 1] if r["pred_1idx"] >= 1 and r["pred_1idx"] - 1 < len(g["steps"]) else "",
    })
json.dump(dump, open(a.dump, "w"), indent=1)
print(f"导出 {len(dump)} 条残余 -> {a.dump}")
