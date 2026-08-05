"""
用官方 GenPRM 推理类在 ProcessBench 上跑,导出每步 reward + 预测首错步。
在远程 GenPRM 环境执行。用法:
  python genprm_run.py --config gsm8k --n 30 --majority 1 --out out_gsm8k.jsonl
"""
import os, sys, json, argparse, time
sys.path.append('/root/autodl-tmp/genprm_work/GenPRM/src')
os.environ['VLLM_USE_V1'] = '0'
from prm_evaluation.genprm_inference import GenPRM, CodeExecutor

SYS = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."
MODEL = "/root/autodl-tmp/genprm_work/models/GenPRM-1.5B"
DATADIR = "/root/autodl-tmp/genprm_work/ProcessBench"

def build_messages(problem, steps):
    sol = "\n\n".join(steps)
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {problem}\n\n{sol}"}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gsm8k")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--majority", type=int, default=1)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--full", action="store_true", help="score ALL steps (no early break) 供 W1/W2 量化")
    ap.add_argument("--out", default="out_gsm8k.jsonl")
    a = ap.parse_args()

    data = json.load(open(f"{DATADIR}/{a.config}.json"))
    # 分层:取前若干 error + correct,保证两桶都有
    err = [d for d in data if d["label"] != -1][: a.n // 2]
    cor = [d for d in data if d["label"] == -1][: a.n - len(err)]
    subset = err + cor
    print(f"loaded {a.config}: {len(subset)} samples ({len(err)} error / {len(cor)} correct)", flush=True)

    # 续跑:已在输出文件里的 id 跳过,追加写
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    todo = [d for d in subset if d["id"] not in done]
    print(f"resume: {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        print("DONE (nothing to run)", flush=True); return

    genprm = GenPRM(MODEL, tensor_parallel_size=1)
    ce = CodeExecutor()
    print("model loaded", flush=True)

    f = open(a.out, "a")
    for c, d in enumerate(todo):
        steps = d["steps"]; msgs = build_messages(d["problem"], steps)
        scores = []; pred = -1
        t0 = time.time()
        for i in range(1, len(steps) + 1):
            try:
                _, reward = genprm.inference(msgs, majority_num=a.majority, cur_step=i,
                                             code_executor=ce, logging=False)
            except Exception as e:
                reward = 0.5
            scores.append(round(float(reward), 4))
            if pred == -1 and reward < a.thr:
                pred = i
                if not a.full:
                    break
        row = {"id": d["id"], "label": d["label"], "n_steps": len(steps),
               "final_answer_correct": d["final_answer_correct"],
               "pred_1idx": pred, "scores": scores, "sec": round(time.time() - t0, 1)}
        f.write(json.dumps(row) + "\n"); f.flush()
        print(f"[{c+1}/{len(todo)}] {d['id']} label={d['label']} pred1={pred} "
              f"nsteps={len(steps)} {row['sec']}s", flush=True)
    f.close()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
