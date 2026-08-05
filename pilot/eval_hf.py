"""transformers 版评测(避开 vLLM tokenizer bug)。用法: python eval_hf.py --model merged_std --config gsm8k --n 120"""
import os, re, json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = "/root/autodl-tmp/genprm_work"
DATADIR = f"{WORK}/ProcessBench"
SYS = "You are a math teacher. Review the solution and judge whether the LAST step is correct."
RESP = "\nJudgement:"

def parse_verdict(g):
    m = re.search(r"boxed\{(Yes|No)\}", g, re.I)
    if m:
        return 1 if m.group(1).lower() == "no" else 0
    gl = g.lower()
    if "incorrect" in gl or "is wrong" in gl or "error" in gl or re.search(r"\bno\b", gl):
        return 1
    if "correct" in gl or re.search(r"\byes\b", gl):
        return 0
    return 0

def build(problem, steps, i):
    users = [problem + "\n" + steps[0]] + steps[1:i]
    sofar = "\n\n".join(users)
    return (f"{SYS}\n\n{sofar}\n\nIs the LAST step above a correct inference/computation? "
            f"Answer with exactly one word: YES or NO.{RESP}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--config", default="gsm8k")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    data = json.load(open(f"{DATADIR}/{a.config}.json"))
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
    model = AutoModelForCausalLM.from_pretrained(f"{WORK}/{a.model}", dtype=torch.float16, device_map="cuda")
    model.eval()
    verd = {}
    for b in range(0, len(prompts), a.bs):
        chunk = prompts[b:b + a.bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1400).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (si, i), g in zip(idx[b:b + a.bs], gen):
            verd[(si, i)] = parse_verdict(g)
        print(f"  ...{b+len(chunk)}/{len(prompts)}", flush=True)

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
