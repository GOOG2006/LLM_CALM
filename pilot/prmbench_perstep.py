"""
我们的机制 = 每步独立验证(去 early-commitment)。与 baseline(顺序找首错, results_prmbench.jsonl)配对对比。
同 seed 同 180 条。每步:给 问题+前缀,单独判该步 YES/NO;首个 NO = 预测错误步(全 YES=0)。
用法: python prmbench_perstep.py [--smoke] [--workers 8] [--cap 60]
"""
import os, re, json, argparse, sys, random, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from openai import OpenAI

client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.environ.get("PILOT_MODEL", "gemini-2.5-flash")

STEP_PROMPT = """You are verifying ONE step of a math solution. Given the problem and all previous steps (assume they are context, not necessarily correct), decide whether THIS step is a valid, correct inference/computation.

Problem:
{q}

Previous steps:
{prev}

STEP TO CHECK:
Step {i}: {step}

Answer with exactly one word: YES (this step is correct) or NO (this step contains an error)."""

def judge_step(q, prev_steps, i, step):
    prev = "\n".join(f"Step {j+1}: {s}" for j, s in enumerate(prev_steps)) or "(none)"
    r = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=512,
        messages=[{"role": "user", "content": STEP_PROMPT.format(q=q, prev=prev, i=i, step=step)}])
    t = (r.choices[0].message.content or "").strip().upper()
    m = re.search(r"\b(YES|NO)\b", t)
    return m.group(1) if m else "YES"   # 无法解析当作 YES(不误报)

def first_error(q, steps, cap):
    for i, s in enumerate(steps[:cap], start=1):
        if judge_step(q, steps[:i-1], i, s) == "NO":
            return i
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=int, default=60)   # 每题最多验前 cap 步(控成本)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_perstep.jsonl"))
    a = ap.parse_args()
    if a.smoke:
        a.per, a.workers = 2, 3

    from datasets import load_dataset
    ds = load_dataset("hitsmy/PRMBench_Preview", split="train")
    by_cls = defaultdict(list)
    for i, c in enumerate(ds["classification"]):
        by_cls[c].append(i)
    random.seed(0)                       # 与 baseline 完全同一抽样
    picks = []
    for c, idxs in by_cls.items():
        random.shuffle(idxs); picks += idxs[:a.per]

    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(i):
        ex = ds[i]; q = ex["question"]; steps = ex["modified_process"]; errs = ex["error_steps"] or []
        pred = first_error(q, steps, a.cap)
        is_cor = len(errs) == 0
        hit = (pred == 0) if is_cor else (pred in errs)
        row = {"idx": ex["idx"], "classification": ex["classification"], "error_steps": errs,
               "pred_step": pred, "n_steps": len(steps), "correct_sample": is_cor, "hit": bool(hit)}
        with lock:
            done[0] += 1
            f.write(json.dumps(row) + "\n"); f.flush()
            if done[0] % 20 == 0:
                print(f"  ...{done[0]}/{len(picks)}", flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex_:
        futs = {ex_.submit(work, i): i for i in picks}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[skip] {futs[fut]} {e}", flush=True)
    f.close()
    compare(rows)

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def stats(rows):
    err = [r for r in rows if not r["correct_sample"] and r["error_steps"]]
    cor = [r for r in rows if r["correct_sample"]]
    rec = sum(r["hit"] for r in err) / max(len(err), 1)
    fp = sum(1 for r in cor if not r["hit"]) / max(len(cor), 1)
    lon = [r for r in err if r["n_steps"] >= 15]; sh = [r for r in err if r["n_steps"] <= 8]
    miss_long = 1 - sum(r["hit"] for r in lon) / max(len(lon), 1)
    miss_short = 1 - sum(r["hit"] for r in sh) / max(len(sh), 1)
    return rec, fp, miss_short, miss_long

def compare(ours):
    base_path = os.path.join(os.path.dirname(__file__), "results_prmbench.jsonl")
    print("\n==================== 机制对比(同 base=gemini-flash, 同 180 条) ====================")
    print(f"{'机制':<26}{'recall':>8}{'误报率':>8}{'短链miss':>10}{'长链miss':>10}")
    if os.path.exists(base_path):
        b = stats(load(base_path))
        print(f"{'baseline 顺序找首错':<24}{b[0]:>8.3f}{b[1]:>8.3f}{b[2]:>10.2f}{b[3]:>10.2f}")
    o = stats(ours)
    print(f"{'ours 每步独立验证':<24}{o[0]:>8.3f}{o[1]:>8.3f}{o[2]:>10.2f}{o[3]:>10.2f}")
    print("判据: ours 长链miss 显著低于 baseline 且误报率不失控 → 机制改进成立")

if __name__ == "__main__":
    main()
