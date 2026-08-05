"""评测 判定-only 验证器(base+LoRA)在 ProcessBench 上。用法: python eval_judge.py --adapter adapter_std --config gsm8k --n 120"""
import os, re, json, argparse
from vllm import LLM, SamplingParams

BASE = "/root/autodl-tmp/genprm_work/models/DSR1-1.5B"
DATADIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SYS = "You are a math teacher. Review the solution and judge whether the LAST step is correct."
RESP = "\nJudgement:"

def build(problem, steps, i):   # 判第 i 步(1-indexed),sofar=题目+step1..stepi
    users = [problem + "\n" + steps[0]] + steps[1:i]
    sofar = "\n\n".join(users)
    return (f"{SYS}\n\n{sofar}\n\nIs the LAST step above a correct inference/computation? "
            f"Answer with exactly one word: YES or NO.{RESP}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--config", default="gsm8k")
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    data = json.load(open(f"{DATADIR}/{a.config}.json"))
    err = [d for d in data if d["label"] != -1][: a.n // 2]
    cor = [d for d in data if d["label"] == -1][: a.n - a.n // 2]
    sub = err + cor

    prompts, idx = [], []   # idx: (sample_i, step 1-indexed)
    for si, d in enumerate(sub):
        for i in range(1, len(d["steps"]) + 1):
            prompts.append(build(d["problem"], d["steps"], i)); idx.append((si, i))

    llm = LLM(model=os.path.join("/root/autodl-tmp/genprm_work", a.model), dtype="half",
              enable_chunked_prefill=False, enforce_eager=False, max_model_len=4096, gpu_memory_utilization=0.9)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6))

    # 每样本每步 verdict
    verd = {}
    for (si, i), o in zip(idx, outs):
        t = o.outputs[0].text.strip().upper()
        m = re.search(r"\b(YES|NO)\b", t)
        verd[(si, i)] = 1 if (m and m.group(1) == "NO") else 0   # 1=判错

    err_hit = err_tot = cor_hit = cor_tot = 0
    for si, d in enumerate(sub):
        n = len(d["steps"])
        pred = next((i - 1 for i in range(1, n + 1) if verd.get((si, i), 0) == 1), -1)  # 0-indexed 首错
        if d["label"] == -1:
            cor_tot += 1; cor_hit += (pred == -1)
        else:
            err_tot += 1; err_hit += (pred == d["label"])
    ae = err_hit / max(err_tot, 1); ac = cor_hit / max(cor_tot, 1)
    f1 = 2 * ae * ac / max(ae + ac, 1e-9)
    print(f"\n===== [{a.model}] on {a.config} (错{err_tot}/对{cor_tot}) =====")
    print(f"error-acc={ae:.3f}  correct-acc={ac:.3f}  F1={f1*100:.1f}")

if __name__ == "__main__":
    main()
