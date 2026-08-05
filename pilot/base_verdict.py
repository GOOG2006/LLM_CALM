"""跑 DSR1-1.5B 零样本判定,标出 base 判错的样本(base_wrong)。用法: python base_verdict.py --n 20000"""
import json, re, argparse, torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append("/root/autodl-tmp/genprm_work")
from eval_hf import parse_verdict
BASE = "/root/autodl-tmp/genprm_work/models/DSR1-1.5B"
DATA = "/root/autodl-tmp/genprm_work/sft_judg.jsonl"

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--bs", type=int, default=48)
a = ap.parse_args()
rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()][: a.n]
tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda").eval()

out = []
for b in range(0, len(rows), a.bs):
    chunk = rows[b:b + a.bs]
    prompts = [r["prompt"][:2600] + "\nJudgement:" for r in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1200).to("cuda")
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
    dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for r, d in zip(chunk, dec):
        bv = "NO" if parse_verdict(d) == 1 else "YES"   # base 判定
        out.append({"prompt": r["prompt"], "label": r["label"], "base": bv,
                    "base_wrong": int(bv != r["label"])})
    if b % (a.bs * 20) == 0:
        print(f"  ...{b+len(chunk)}/{len(rows)}", flush=True)

json.dump(out, open("/root/autodl-tmp/genprm_work/base_verdict.json", "w"))
bw = sum(r["base_wrong"] for r in out)
# 分类看: 错步里 base 漏了多少(base=YES 但 gold=NO)
miss_err = sum(1 for r in out if r["label"] == "NO" and r["base"] == "YES")
print(f"N={len(out)} base判错={bw} ({100*bw/len(out):.1f}%)  错步中base漏检(判YES)={miss_err}")
