"""
对标 2026 baseline。共享 stage-1(每步 CoT 批判),ours = baseline + 全局确认层。
机制:
  perstep_cot = 每步 CoT 批判(2026 GenRM 式强 baseline),预测=第一个 ERROR
  ours        = 上述批判当 nominate(收集所有 ERROR 候选) → 每候选全局确认 → 预测=第一个 CONFIRMED
  (SC 投票 / 朴素顺序 从 results_compare.jsonl 按 idx 并入,同样本)
判据:ours FP < perstep_cot 且 recall 不降 → 全局确认层成立。
用法: python prmbench_v2.py [--smoke] [--per 20 --ncorrect 100 --cap 60 --workers 10]
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

CRITIQUE = """You are verifying ONE step of a math solution. Given the problem and previous steps (context, may contain errors), reason briefly whether THIS step is a valid, correct inference/computation, then give a verdict.

Problem:
{q}

Previous steps:
{prev}

STEP TO CHECK:
Step {i}: {step}

Reason in 1-2 sentences, then end with exactly: 'Verdict: CORRECT' or 'Verdict: ERROR'."""

CONFIRM = """A step-checker flagged one step of the solution below as a possible error. Read the ENTIRE solution (later steps may resolve or contextualize it), then decide whether the flagged step is a GENUINE error that actually makes the reasoning wrong, or a false alarm.

Problem:
{q}

Full solution:
{full}

FLAGGED Step {i}: {step}

Reason in 1-2 sentences, then end with exactly: 'Verdict: CONFIRMED' or 'Verdict: SPURIOUS'."""

def chat(content, mx=640):
    r = client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=mx,
                                       messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content or ""

def critique_step(q, steps, i):
    prev = "\n".join(f"Step {j+1}: {s}" for j, s in enumerate(steps[:i-1])) or "(none)"
    out = chat(CRITIQUE.format(q=q, prev=prev, i=i, step=steps[i-1]))
    m = re.search(r"Verdict:\s*(CORRECT|ERROR)", out, re.I)
    return (m.group(1).upper() == "ERROR") if m else False

def confirm_step(q, steps, i):
    full = "\n".join(f"Step {j+1}: {s}" for j, s in enumerate(steps))
    out = chat(CONFIRM.format(q=q, full=full, i=i, step=steps[i-1]))
    m = re.search(r"Verdict:\s*(CONFIRMED|SPURIOUS)", out, re.I)
    return (m.group(1).upper() == "CONFIRMED") if m else True  # 无法解析当确认(不主动降recall)

def hit_of(pred, errs, is_cor):
    return (pred == 0) if is_cor else (pred in errs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--ncorrect", type=int, default=100)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_v2.jsonl"))
    a = ap.parse_args()
    if a.smoke:
        a.per, a.ncorrect, a.workers = 2, 4, 3

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
        n = min(len(steps), a.cap)
        # stage-1: 每步 CoT 批判(共享)
        flagged = [k for k in range(1, n + 1) if critique_step(q, steps, k)]
        calls = n
        base_pred = flagged[0] if flagged else 0          # baseline: 第一个 ERROR
        # stage-2: 全局确认(仅候选)
        ours_pred = 0
        for k in flagged:
            calls += 1
            if confirm_step(q, steps, k):
                ours_pred = k; break
        row = {"idx": ex["idx"], "classification": ex["classification"], "error_steps": errs,
               "n_steps": len(steps), "correct_sample": is_cor, "calls": calls,
               "perstep_cot": {"pred": base_pred, "hit": hit_of(base_pred, errs, is_cor)},
               "ours": {"pred": ours_pred, "hit": hit_of(ours_pred, errs, is_cor)}}
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
    # 并入上轮 SC / 朴素顺序
    prev = {}
    p = os.path.join(os.path.dirname(__file__), "results_compare.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); prev[r["idx"]] = r
    err = [r for r in rows if not r["correct_sample"] and r["error_steps"]]
    cor = [r for r in rows if r["correct_sample"]]
    lon = [r for r in err if r["n_steps"] >= 15]; sh = [r for r in err if r["n_steps"] <= 8]
    avg_calls = sum(r["calls"] for r in rows) / max(len(rows), 1)
    print(f"\n===== 对标2026 (base=gemini-flash) 错误N={len(err)} 全对N={len(cor)} 长链N={len(lon)} =====")
    print(f"{'机制':<24}{'recall':>8}{'误报率':>8}{'长链miss':>10}{'平衡acc':>9}{'调用/题':>8}")

    def line(name, getter, calls):
        rec = sum(getter(r)["hit"] for r in err) / max(len(err), 1)
        fp = sum(1 for r in cor if not getter(r)["hit"]) / max(len(cor), 1)
        ml = 1 - sum(getter(r)["hit"] for r in lon) / max(len(lon), 1)
        bal = (rec + (1 - fp)) / 2
        print(f"{name:<24}{rec:>8.3f}{fp:>8.3f}{ml:>10.2f}{bal:>9.3f}{calls:>8}")

    if prev:
        line("SC投票(2022下限)", lambda r: prev[r["idx"]]["sc_vote"], "5")
    line("每步CoT批判(2026base)", lambda r: r["perstep_cot"], f"{avg_calls-1:.0f}")
    line("ours 提名→全局确认", lambda r: r["ours"], f"{avg_calls:.0f}")
    print("判据: ours 平衡acc/长链miss 优于【每步CoT批判】且 FP 更低 → 确认层成立、机制胜出。")

if __name__ == "__main__":
    main()
