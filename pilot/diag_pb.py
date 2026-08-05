"""从官方 result 文件里,找错误样本中 GenPRM 失败的,打印它在金标错误步上的真实输出。
用法: python diag_pb.py <split_out_dir> [idd=1] [max=6]"""
import json, os, sys, glob, re
sys.stdout.reconfigure(encoding="utf-8")
split_out = sys.argv[1]
idd = sys.argv[2] if len(sys.argv) > 2 else "1"
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 6

folders = sorted(glob.glob(os.path.join(split_out, "*_analyze_verify_execute")))
fails = []
for fo in folders:
    p = os.path.join(fo, f"result_{idd}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    label = d["label"]
    if label == -1:
        continue  # 只看错误样本
    val = d["value"]
    pred = next((k for k, s in enumerate(val) if s < 0.5), -1)  # 0-indexed 首个<0.5
    if pred == label:
        continue  # 抓对了,跳过
    asst = [m["content"] for m in d["conversation"] if m.get("role") == "assistant"]
    fails.append((os.path.basename(fo), label, pred, val, d.get("problem", ""), d.get("steps", []), asst))

print(f"失败的错误样本: {len(fails)}  (只看 error 样本中 pred≠label 的)")
kinds = {"missed(判全对)": 0, "misloc(定位错)": 0}
for f in fails:
    kinds["missed(判全对)" if f[2] == -1 else "misloc(定位错)"] += 1
print("类型:", kinds, "\n")

for name, label, pred, val, prob, steps, asst in fails[:mx]:
    print("#" * 78)
    print(f"{name} | 金标首错步(0idx)={label} 分={val[label] if label < len(val) else 'NA'} | 预测={pred} | 全部分数={[round(x,2) for x in val]}")
    print(f"[金标错误步文本] {steps[label][:260] if label < len(steps) else ''}")
    if label < len(asst):
        print(f"[GenPRM 对该步的实际判断]\n{asst[label][:1400]}")
    print()
