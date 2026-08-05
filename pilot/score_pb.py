"""按 ProcessBench 标准从官方输出算 F1,可聚合多个 idd(Maj@N)。
用法: python score_pb.py <split_out_dir> <idd_list 如 1 或 1,2,..,8>"""
import json, os, sys, glob
split_out, idds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]

folders = sorted(glob.glob(os.path.join(split_out, "*_analyze_verify_execute")))
err_hit = err_tot = cor_hit = cor_tot = 0
missing = 0
for fo in folders:
    # 聚合 idd 的每步 reward(平均)
    per_idd = []
    for idd in idds:
        p = os.path.join(fo, f"result_{idd}.json")
        if os.path.exists(p):
            per_idd.append(json.load(open(p)))
    if not per_idd:
        missing += 1; continue
    label = per_idd[0]["label"]
    n_steps = len(per_idd[0]["value"])
    avg = [sum(r["value"][k] for r in per_idd) / len(per_idd) for k in range(n_steps)]
    pred = next((k for k, s in enumerate(avg) if s < 0.5), -1)  # 0-indexed 首个<0.5,否则-1
    if label == -1:
        cor_tot += 1; cor_hit += (pred == -1)
    else:
        err_tot += 1; err_hit += (pred == label)

acc_err = err_hit / max(err_tot, 1)
acc_cor = cor_hit / max(cor_tot, 1)
f1 = 2 * acc_err * acc_cor / max(acc_err + acc_cor, 1e-9)
print(f"folders={len(folders)} missing_result={missing}  idd={idds}")
print(f"error-acc  = {acc_err:.3f}  ({err_hit}/{err_tot})")
print(f"correct-acc= {acc_cor:.3f}  ({cor_hit}/{cor_tot})")
print(f"ProcessBench-F1 = {f1*100:.1f}   (论文 GenPRM-1.5B GSM8K: Pass@1=52.8, Maj@8=51.3)")
