"""评测 CoT-SFT 验证器。生成 rationale,解析 boxed Yes/No。用法: python eval_cot.py --model merged_cot --config gsm8k --n 100"""
import os, re, json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
WORK = "/root/autodl-tmp/genprm_work"
SYS = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."

def build(problem, steps, i):
    users = [problem + "\n" + steps[0]] + steps[1:i]
    sofar = "\n\n".join(users)
    return (f"{SYS}\n\nQuestion & solution so far:\n{sofar}\n\n"
            f"Critique the LAST paragraph. Reason with analysis and python code, "
            f"then end with **Judgement**: $\\boxed{{Yes}}$ or $\\boxed{{No}}$.")

def parse(g):
    m = re.findall(r"boxed\{(Yes|No)\}", g, re.I)
    if m:
        return 1 if m[-1].lower() == "no" else 0
    gl = g.lower()
    if "incorrect" in gl or re.search(r"\bno\b", gl):
        return 1
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--config", default="gsm8k")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--bs", type=int, default=12)
    a = ap.parse_args()
    data = json.load(open(f"{WORK}/ProcessBench/{a.config}.json"))
    err = [d for d in data if d["label"] != -1][: a.n // 2]
    cor = [d for d in data if d["label"] == -1][: a.n - a.n // 2]
    sub = err + cor
    prompts, idx = [], []
    for si, d in enumerate(sub):
        for i in range(1, len(d["steps"]) + 1):
            prompts.append(build(d["problem"], d["steps"], i)[:3000]); idx.append((si, i))
    tok = AutoTokenizer.from_pretrained(f"{WORK}/{a.model}"); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"{WORK}/{a.model}", dtype=torch.float16, device_map="cuda").eval()
    verd = {}
    for b in range(0, len(prompts), a.bs):
        enc = tok(prompts[b:b + a.bs], return_tensors="pt", padding=True, truncation=True, max_length=1400).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (si, i), g in zip(idx[b:b + a.bs], gen):
            verd[(si, i)] = parse(g)
        print(f"  ...{b+len(prompts[b:b+a.bs])}/{len(prompts)}", flush=True)
    eh = et = ch = ct = 0
    for si, d in enumerate(sub):
        n = len(d["steps"])
        pred = next((i - 1 for i in range(1, n + 1) if verd.get((si, i), 0) == 1), -1)
        if d["label"] == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == d["label"])
    ae = eh / max(et, 1); ac = ch / max(ct, 1); f1 = 2 * ae * ac / max(ae + ac, 1e-9)
    print(f"\n===== [{a.model}] {a.config} (错{et}/对{ct}) error-acc={ae:.3f} correct-acc={ac:.3f} F1={f1*100:.1f}", flush=True)

if __name__ == "__main__":
    main()
