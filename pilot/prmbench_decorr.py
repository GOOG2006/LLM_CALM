"""
验证"去相关验证"框架的 generality:PRMBench 上逐错误类别比
  correlated(读解答理由判) vs decorrelated(先从题面独立重导再比对)。
判据:decorrelated 跨多数类别提召回且 FP 不失控 → 原理成立。
用法: python prmbench_decorr.py [--smoke] [--per 12 --ncorrect 40 --workers 10]
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

CORR = """You are verifying ONE step of a math solution.
Problem:
{q}

Previous steps:
{prev}

STEP TO CHECK:
Step {i}: {step}

Is this step a valid, correct inference/computation? Reason in 1-2 sentences, then end with exactly 'Verdict: CORRECT' or 'Verdict: ERROR'."""

DECORR = """You are verifying ONE step of a math solution by INDEPENDENT re-derivation.
Problem:
{q}

Results established by previous steps:
{prev}

Do NOT assume the step below is correct. FIRST, using ONLY the problem statement and the previous results, independently work out what this step should compute or conclude. THEN compare your independent result to what the step actually claims.

STEP TO CHECK:
Step {i}: {step}

Reason in 1-2 sentences (state your independent value, then compare), then end with exactly 'Verdict: CORRECT' or 'Verdict: ERROR'."""

def chat(content):
    r = client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=640,
                                       messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content or ""

def verdict(text):
    m = re.search(r"Verdict:\s*(CORRECT|ERROR)", text, re.I)
    return (m.group(1).upper() == "ERROR") if m else False  # 无解析当 CORRECT(不误报)

def first_error(tmpl, q, steps, cap):
    for i, s in enumerate(steps[:cap], start=1):
        prev = "\n".join(f"Step {j+1}: {t}" for j, t in enumerate(steps[:i-1])) or "(none)"
        if verdict(chat(tmpl.format(q=q, prev=prev, i=i, step=s))):
            return i
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--ncorrect", type=int, default=40)
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_decorr.jsonl"))
    a = ap.parse_args()
    if a.smoke:
        a.per, a.ncorrect, a.workers = 2, 6, 3

    from datasets import load_dataset
    ds = load_dataset("hitsmy/PRMBench_Preview", split="train")
    by_cls = defaultdict(list)
    for i, c in enumerate(ds["classification"]):
        by_cls[c].append(i)
    random.seed(0)
    picks = []
    for c, idxs in by_cls.items():
        random.shuffle(idxs)
        picks += idxs[:(a.ncorrect if c == "multi_solutions" else a.per)]

    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(i):
        ex = ds[i]; q = ex["question"]; steps = ex["modified_process"]; errs = ex["error_steps"] or []
        is_cor = len(errs) == 0
        p_corr = first_error(CORR, q, steps, a.cap)
        p_deco = first_error(DECORR, q, steps, a.cap)
        row = {"idx": ex["idx"], "classification": ex["classification"], "error_steps": errs,
               "n_steps": len(steps), "correct_sample": is_cor,
               "corr": {"pred": p_corr, "hit": (p_corr == 0) if is_cor else (p_corr in errs)},
               "deco": {"pred": p_deco, "hit": (p_deco == 0) if is_cor else (p_deco in errs)}}
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
    report(rows)

def report(rows):
    err = [r for r in rows if not r["correct_sample"] and r["error_steps"]]
    cor = [r for r in rows if r["correct_sample"]]
    print(f"\n===== 去相关验证 generality (base=gemini-flash) 错误N={len(err)} 全对N={len(cor)} =====")
    print(f"{'错误类别':<22}{'corr召回':>9}{'deco召回':>9}{'Δ':>7}")
    by = defaultdict(list)
    for r in err:
        by[r["classification"]].append(r)
    wins = 0
    for c in sorted(by):
        g = by[c]
        rc = sum(x["corr"]["hit"] for x in g) / len(g)
        rd = sum(x["deco"]["hit"] for x in g) / len(g)
        d = rd - rc
        wins += (d > 0)
        print(f"{c:<22}{rc:>9.2f}{rd:>9.2f}{d:>+7.2f}{'  ✓' if d > 0 else ''}")
    fc = sum(1 for r in cor if not r["corr"]["hit"]) / max(len(cor), 1)
    fd = sum(1 for r in cor if not r["deco"]["hit"]) / max(len(cor), 1)
    rc_all = sum(x["corr"]["hit"] for x in err) / max(len(err), 1)
    rd_all = sum(x["deco"]["hit"] for x in err) / max(len(err), 1)
    print(f"{'--- 总计 ---':<22}{rc_all:>9.2f}{rd_all:>9.2f}{rd_all-rc_all:>+7.2f}")
    print(f"FP(全对样本): corr={fc:.2f}  deco={fd:.2f}")
    print(f"deco 提升的类别数: {wins}/{len(by)}  → {'跨多数类别成立' if wins > len(by)/2 else '太窄,不成立'}")

if __name__ == "__main__":
    main()
