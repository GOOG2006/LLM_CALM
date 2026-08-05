"""抽 CoT-rationale 按步 SFT 数据: (题目+步骤 -> 该步完整 analyze/verify/判定 rationale)。"""
import pandas as pd, re, json, sys
sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("/root/autodl-tmp/genprm_work/GenPRM-Data/data/train-00000-of-00001.parquet")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SYS = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."

out = []
for idx in range(min(N, len(df))):
    conv = df.iloc[idx]["conversations"]
    users, rationales, labels = [], [], []
    for m in conv:
        if m["role"] == "user":
            users.append(m["content"])
        elif m["role"] == "assistant":
            rationales.append(m["content"])
            j = re.search(r"boxed\{(Yes|No)\}", m["content"])
            labels.append("YES" if (j and j.group(1) == "Yes") else "NO")
    for k in range(min(len(users), len(rationales), len(labels))):
        sofar = "\n\n".join(users[: k + 1])
        prompt = (f"{SYS}\n\nQuestion & solution so far:\n{sofar}\n\n"
                  f"Critique the LAST paragraph. Reason with analysis and python code, "
                  f"then end with **Judgement**: $\\boxed{{Yes}}$ or $\\boxed{{No}}$.")
        out.append({"prompt": prompt, "rationale": rationales[k], "label": labels[k]})

with open("/root/autodl-tmp/genprm_work/sft_cot.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
yes = sum(1 for r in out if r["label"] == "YES")
avglen = sum(len(r["rationale"]) for r in out) // len(out)
print(f"抽出 {len(out)} 例; YES={yes} NO={len(out)-yes} 错步{100*(len(out)-yes)/len(out):.1f}%; rationale 平均{avglen}字符")
