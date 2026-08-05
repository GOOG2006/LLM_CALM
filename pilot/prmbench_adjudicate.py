"""对某 classification 的漏检案例(hit=false),打印错误步文本+reason,人工裁决是否真错。不发 API。"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from datasets import load_dataset

cls = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows = [json.loads(l) for l in open(
    "C:/Users/xjtuy/LLM_Result_Validation/pilot/results_prmbench.jsonl", encoding="utf-8") if l.strip()]
missed = [r for r in rows if r["classification"] == cls and not r["hit"] and not r["correct_sample"]]

ds = load_dataset("hitsmy/PRMBench_Preview", split="train")
by_idx = {ds[i]["idx"]: i for i in range(len(ds))}

print(f"=== classification={cls}  漏检 {len(missed)} 条,展示 {min(k,len(missed))} ===\n")
for r in missed[:k]:
    ex = ds[by_idx[r["idx"]]]
    steps = ex["modified_process"]; errs = ex["error_steps"]
    print("#" * 70)
    print(f"idx={r['idx']}  error_steps={errs}  模型预测={r['pred_step']}")
    print(f"[问题] {ex['question'][:200]}")
    for s in errs:
        if 1 <= s <= len(steps):
            print(f"[错误步 {s}] {steps[s-1]}")
    print(f"[PRMBench reason] {ex['reason'][:400]}")
    print()
