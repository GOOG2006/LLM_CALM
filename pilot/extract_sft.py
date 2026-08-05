"""从 GenPRM 对话数据抽 判定-only SFT 例子: (题目+步骤 -> 该步 Yes/No)。"""
import pandas as pd, re, json, sys
sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("/root/autodl-tmp/genprm_work/GenPRM-Data/data/train-00000-of-00001.parquet")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

SYS = "You are a math teacher. Review the solution and judge whether the LAST step is correct."
out = []
for idx in range(min(N, len(df))):
    conv = df.iloc[idx]["conversations"]
    users, labels = [], []
    for m in conv:
        if m["role"] == "user":
            users.append(m["content"])
        elif m["role"] == "assistant":
            j = re.search(r"boxed\{(Yes|No)\}", m["content"])
            labels.append("Yes" if (j and j.group(1) == "Yes") else "No")
    # user[0] = 题目+步骤1; user[k] = 步骤k+1; 每个 assistant 判 user[k]
    for k in range(min(len(users), len(labels))):
        sofar = "\n\n".join(users[: k + 1])
        prompt = (f"{SYS}\n\n{sofar}\n\nIs the LAST step above a correct inference/computation? "
                  f"Answer with exactly one word: YES or NO.")
        out.append({"prompt": prompt, "label": "YES" if labels[k] == "Yes" else "NO"})

with open("/root/autodl-tmp/genprm_work/sft_judg.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
yes = sum(1 for r in out if r["label"] == "YES")
print(f"抽出 {len(out)} 例 (来自 {min(N,len(df))} 对话); YES={yes} NO={len(out)-yes} "
      f"错步占比 {100*(len(out)-yes)/len(out):.1f}%")
print("--- 一例 prompt(截断) ---"); print(out[0]["prompt"][:400]); print("label=", out[0]["label"])
