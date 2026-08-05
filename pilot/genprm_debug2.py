"""对指定 id 的金标错误步,用 默认 vs 改进 verify_template 各跑一次,看 GenPRM 代码是否改为从题面独立重算、以及分数变化。
用法: python genprm_debug2.py --config gsm8k --ids gsm8k-8"""
import os, sys, json, argparse
sys.path.append('/root/autodl-tmp/genprm_work/GenPRM/src')
os.environ['VLLM_USE_V1'] = '0'
from prm_evaluation.genprm_inference import GenPRM, CodeExecutor

SYS = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."
MODEL = "/root/autodl-tmp/genprm_work/models/GenPRM-1.5B"
DATADIR = "/root/autodl-tmp/genprm_work/ProcessBench"

DEFAULT_VT = "<verify>\nLet's use python code to find any potential error:\n```python\n"
IMPROVED_VT = ("<verify>\nLet me recompute this step's result directly from the quantities stated in the "
               "ORIGINAL problem, WITHOUT reusing the variables or substitutions from the student's solution:\n```python\n")

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="gsm8k")
ap.add_argument("--ids", required=True)
a = ap.parse_args()
data = {d["id"]: d for d in json.load(open(f"{DATADIR}/{a.config}.json"))}
genprm = GenPRM(MODEL, tensor_parallel_size=1)
ce = CodeExecutor()

for _id in a.ids.split(","):
    d = data[_id]; steps = d["steps"]; gi = d["label"]
    # 官方多轮格式
    di = list(steps); di[0] = d["problem"] + "\n" + di[0]
    conv = [{"role": "system", "content": SYS}]
    for s in di:
        conv.append({"role": "user", "content": s}); conv.append({"role": "assistant", "content": ""})
    # paths = 到金标错误步(1-indexed = gi+1)的 assistant 位置前
    cur = gi + 1
    asst_positions = [k for k, m in enumerate(conv) if m["role"] == "assistant"]
    paths = conv[: asst_positions[gi]]
    print("#" * 78); print(f"id={_id} 金标错误步(0idx)={gi} => step{cur}")
    print(f"[错误步] {steps[gi][:200]}")
    for name, vt in [("默认模板", DEFAULT_VT), ("改进模板(题面独立重算)", IMPROVED_VT)]:
        outs, reward = genprm.inference(paths, majority_num=1, cur_step=cur, code_executor=ce,
                                        verify_template=vt, logging=False)
        print(f"\n===== {name}  reward={reward:.3f} =====")
        print(outs[0][:1600])
