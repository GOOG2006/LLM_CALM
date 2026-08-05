"""
独立重导验证器(结构性去相关)。对 GenPRM 已跑的同一批 ProcessBench 样本:
  每步:让模型只据 题目+前步 独立重算该步应得的数(禁止沿用步骤算术)K 次 -> 多数 m;
  离群检验:m 与步骤实际数不符 -> 该步为错(首个即预测)。判决=数值比较,不经 CoT Yes/No。
对照 GenPRM(genprm_preds.json)。
用法: python ind_verify.py [--k 5 --workers 10 --n 100]
"""
import os, re, json, argparse, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.environ.get("PILOT_MODEL", "gemini-2.5-flash")

PROMPT = """Problem:
{q}

Work so far (previous steps, assume they are correct context):
{prev}

Below is the NEXT step written by a student. It computes some quantity.
Student step: {step}

Task: Independently compute the final numerical value that this step SHOULD arrive at, using ONLY the problem and the previous steps. Do NOT reuse or trust the arithmetic, substitutions, or numbers chosen inside the student step — derive the value yourself from the problem. Output ONLY the final number (digits, optionally a decimal), nothing else."""

def last_num(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None

def recompute(q, prev, step, k):
    vals = []
    for _ in range(k):
        r = client.chat.completions.create(model=MODEL, temperature=0.8, max_tokens=512,
              messages=[{"role": "user", "content": PROMPT.format(q=q, prev=prev, step=step)}])
        vals.append(last_num(r.choices[0].message.content))
    return vals

def eq(a, b):
    if a is None or b is None:
        return True  # 无法判定则不 flag(不误报)
    return abs(a - b) <= 1e-6 + 1e-3 * abs(b)

def first_error(q, steps, k):
    for i, s in enumerate(steps):
        prev = "\n".join(f"Step {j+1}: {t}" for j, t in enumerate(steps[:i])) or "(none)"
        gi = last_num(s)
        if gi is None:
            continue
        vals = [v for v in recompute(q, prev, s, k) if v is not None]
        if not vals:
            continue
        m = Counter([round(v, 4) for v in vals]).most_common(1)[0][0]  # 多数独立值
        if not eq(m, gi):
            return i  # 0-indexed 首个离群步
    return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_ind.jsonl"))
    a = ap.parse_args()

    data = json.load(open(os.path.join(os.path.dirname(__file__), "genprm_preds.json"), encoding="utf-8"))[: a.n]
    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(d):
        pred = first_error(d["problem"], d["steps"], a.k)
        row = {"name": d["name"], "label": d["label"], "genprm_pred": d["genprm_pred"], "ind_pred": pred}
        with lock:
            done[0] += 1
            f.write(json.dumps(row) + "\n"); f.flush()
            if done[0] % 10 == 0:
                print(f"  ...{done[0]}/{len(data)}", flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, d) for d in data]
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[skip] {e}", flush=True)
    f.close()
    report(rows)

def report(rows):
    err = [r for r in rows if r["label"] != -1]
    cor = [r for r in rows if r["label"] == -1]
    def acc_e(key): return sum(r[key] == r["label"] for r in err) / max(len(err), 1)
    def acc_c(key): return sum(r[key] == -1 for r in cor) / max(len(cor), 1)
    print(f"\n===== 独立验证器 vs GenPRM (同 {len(rows)} 样本, 错{len(err)}/对{len(cor)}) =====")
    for key, name in [("genprm_pred", "GenPRM(1.5B官方)"), ("ind_pred", "独立重导+离群(ours)")]:
        ae, ac = acc_e(key), acc_c(key)
        f1 = 2 * ae * ac / max(ae + ac, 1e-9)
        print(f"{name:<22} error-acc={ae:.3f}  correct-acc={ac:.3f}  F1={f1*100:.1f}")
    # 关键:GenPRM 漏的错,ours 捞回多少
    genprm_missed = [r for r in err if r["genprm_pred"] != r["label"]]
    ours_rescue = [r for r in genprm_missed if r["ind_pred"] == r["label"]]
    print(f"\nGenPRM 漏检/错定位的错误样本: {len(genprm_missed)}")
    print(f"其中 ours 捞回(定位对): {len(ours_rescue)}  ({len(ours_rescue)/max(len(genprm_missed),1)*100:.0f}%)")
    # ours 新引入的误报(对样本被 ours 误判)
    new_fp = [r for r in cor if r["ind_pred"] != -1 and r["genprm_pred"] == -1]
    print(f"ours 新增误报(GenPRM 判对、ours 误判的对样本): {len(new_fp)}/{len(cor)}")

if __name__ == "__main__":
    main()
