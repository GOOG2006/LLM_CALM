"""transformers 版 带代码执行 的 CoT 评测(复刻 GenPRM 流程,绕开 vLLM)。
每步: 生成 analyze+code -> 真执行 code -> 注入真实输出 -> 再生成判定。用法: python eval_cot_exec_hf.py --model merged_cot --n 40"""
import os, re, json, argparse, torch, sys
sys.path.append("/root/autodl-tmp/genprm_work/GenPRM/src")
from prm_evaluation.genprm_inference import CodeExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
WORK = "/root/autodl-tmp/genprm_work"
SYS = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."

def build(problem, steps, i):
    users = [problem + "\n" + steps[0]] + steps[1:i]
    sofar = "\n\n".join(users)
    return (f"{SYS}\n\nQuestion & solution so far:\n{sofar}\n\n"
            f"Critique the LAST paragraph. Reason with analysis and python code, "
            f"then end with **Judgement**: $\\boxed{{Yes}}$ or $\\boxed{{No}}$.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--config", default="gsm8k"); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    data = json.load(open(f"{WORK}/ProcessBench/{a.config}.json"))
    err = [d for d in data if d["label"] != -1][: a.n // 2]
    cor = [d for d in data if d["label"] == -1][: a.n - a.n // 2]
    sub = err + cor
    tok = AutoTokenizer.from_pretrained(f"{WORK}/{a.model}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"{WORK}/{a.model}", dtype=torch.float16, device_map="cuda").eval()
    ce = CodeExecutor()

    def gen(text, maxnew):
        e = tok(text, return_tensors="pt", truncation=True, max_length=1600).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(o[0, e["input_ids"].shape[1]:], skip_special_tokens=True)

    def judge_step(problem, steps, i):
        p = build(problem, steps, i)[:3000]
        g1 = gen(p, 300)
        m = re.search(r"```python\s*(.*?)```", g1, re.DOTALL)
        if m:
            code = m.group(1)
            try:
                out = ce.execute("```python\n" + code + "\n```")
            except Exception:
                out = ""
            pre = g1[: m.end()]   # 到代码块结束
            cont = p + pre + f"\n[Code Output]\n```\n{out}\n```\n</verify>\n<output>\n**Judgement**: $\\boxed{{"
            g2 = gen(cont, 4)
            mm = re.search(r"(Yes|No)", g2)
            if mm:
                return 1 if mm.group(1) == "No" else 0
        m2 = re.findall(r"boxed\{(Yes|No)\}", g1, re.I)
        return 1 if (m2 and m2[-1].lower() == "no") else 0

    eh = et = ch = ct = 0
    for si, d in enumerate(sub):
        n = len(d["steps"])
        pred = -1
        for i in range(1, n + 1):
            if judge_step(d["problem"], d["steps"], i) == 1:
                pred = i - 1; break
        if d["label"] == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == d["label"])
        print(f"  [{si+1}/{len(sub)}] label={d['label']} pred={pred}", flush=True)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); f1 = 2 * ae * ac / max(ae + ac, 1e-9)
    print(f"\n===== [{a.model}+EXEC] (错{et}/对{ct}) error-acc={ae:.3f} correct-acc={ac:.3f} F1={f1*100:.1f}", flush=True)

if __name__ == "__main__":
    main()
