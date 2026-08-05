"""对指定 id 打印 题目 + gold + 一条低温推理链,人工看'自信错'错在哪。"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.environ.get("BASE_MODEL", "gemini-2.5-flash-lite")
SYS = "Solve the math problem step by step. End with a line 'Answer: <number>'."
ds = load_dataset("openai/gsm8k", "main", split="test")

def num(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
    return m[-1] if m else None

for ix in [int(x) for x in sys.argv[1:]]:
    q = ds[ix]["question"]; gold = ds[ix]["answer"]
    r = client.chat.completions.create(model=MODEL, temperature=0.2, max_tokens=1024,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": q}])
    sol = r.choices[0].message.content
    print("=" * 78)
    print(f"id={ix}")
    print("[题目]", q)
    print("[GOLD 解]", gold.strip())
    print("[模型解]", sol.strip())
    print()
