"""对指定 id 打印 GenPRM 每步完整输出(analyze/verify/code)+ reward,看它为何放过真错。
用法: python genprm_debug.py --config gsm8k --ids gsm8k-8,gsm8k-9 --majority 1 --focus_step N(仅打印该步)"""
import os, sys, json, argparse
sys.path.append('/root/autodl-tmp/genprm_work/GenPRM/src')
os.environ['VLLM_USE_V1'] = '0'
from prm_evaluation.genprm_inference import GenPRM, CodeExecutor

SYS = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."
MODEL = "/root/autodl-tmp/genprm_work/models/GenPRM-1.5B"
DATADIR = "/root/autodl-tmp/genprm_work/ProcessBench"

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="gsm8k")
ap.add_argument("--ids", required=True)
ap.add_argument("--majority", type=int, default=1)
a = ap.parse_args()

data = {d["id"]: d for d in json.load(open(f"{DATADIR}/{a.config}.json"))}
genprm = GenPRM(MODEL, tensor_parallel_size=1)
ce = CodeExecutor()

for _id in a.ids.split(","):
    d = data[_id]; steps = d["steps"]
    sol = "\n\n".join(steps)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {d['problem']}\n\n{sol}"}]
    gi = d["label"]  # 0-indexed 首错步
    print("\n" + "#" * 80)
    print(f"id={_id} gold首错步(0idx)={gi} => 关注 1-indexed step {gi+1}")
    print("[问题]", d["problem"])
    print(f"[真错步文本 step{gi+1}]", steps[gi])
    # 只跑到并打印 gold 错误步的完整验证
    for i in range(1, len(steps) + 1):
        outs, reward = genprm.inference(msgs, majority_num=a.majority, cur_step=i,
                                        code_executor=ce, logging=False)
        if i == gi + 1:
            print(f"\n===== GenPRM 对真错步 step{i} 的完整输出 (reward={reward:.3f}) =====")
            print(outs[0][:2500])
            break
        else:
            print(f"  step{i}: reward={reward:.3f}")
