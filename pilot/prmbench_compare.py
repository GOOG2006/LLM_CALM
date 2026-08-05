"""
三机制配对对比(同 base=gemini-flash,同一批样本)。见 FINDINGS_prmbench.md。
机制:
  seq1     = 朴素顺序找首错(单遍, temp0)
  sc_vote  = 顺序找首错 K 次(temp0.7)多数投票  ← 强 baseline
  perstep  = 每步独立验证(temp0, cap)          ← ours
样本:8 错误类 × per + multi_solutions × n_correct(扩样测 FP)。
用法: python prmbench_compare.py [--smoke] [--per 20 --ncorrect 100 --ksc 5 --workers 10 --cap 60]
"""
import os, re, json, argparse, sys, random, threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from openai import OpenAI

client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.environ.get("PILOT_MODEL", "gemini-2.5-flash")

FIND_PROMPT = """You are a careful math reasoning verifier. Below is a problem and a step-by-step solution with numbered steps. Identify the FIRST step that contains an error (mathematical, logical, or unfounded claim). If EVERY step is correct, answer 0.

Problem:
{q}

Solution steps:
{steps}

Think briefly, then end with a line exactly: 'First error step: <N>' (0 if no error)."""

STEP_PROMPT = """You are verifying ONE step of a math solution. Given the problem and previous steps (context, not necessarily correct), decide whether THIS step is a valid, correct inference/computation.

Problem:
{q}

Previous steps:
{prev}

STEP TO CHECK:
Step {i}: {step}

Answer with exactly one word: YES or NO."""

def chat(content, temp, mx=1024):
    r = client.chat.completions.create(model=MODEL, temperature=temp, max_tokens=mx,
                                       messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content or ""

def parse_first(text, n):
    m = re.search(r"First error step:\s*(\d+)", text)
    if m:
        v = int(m.group(1)); return v if 0 <= v <= n else None
    nums = re.findall(r"\b(\d+)\b", text)
    return int(nums[-1]) if nums else None

def find_first(q, steps, temp):
    numbered = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    return parse_first(chat(FIND_PROMPT.format(q=q, steps=numbered), temp), len(steps))

def sc_vote(q, steps, k):
    preds = [find_first(q, steps, 0.7) for _ in range(k)]
    preds = [p for p in preds if p is not None]
    if not preds:
        return 0
    return Counter(preds).most_common(1)[0][0]

def perstep(q, steps, cap):
    for i, s in enumerate(steps[:cap], start=1):
        prev = "\n".join(f"Step {j+1}: {t}" for j, t in enumerate(steps[:i-1])) or "(none)"
        out = chat(STEP_PROMPT.format(q=q, prev=prev, i=i, step=s), 0.0, 512).strip().upper()
        m = re.search(r"\b(YES|NO)\b", out)
        if m and m.group(1) == "NO":
            return i
    return 0

def hit_of(pred, errs, is_cor):
    return (pred == 0) if is_cor else (pred in errs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--ncorrect", type=int, default=100)
    ap.add_argument("--ksc", type=int, default=5)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_compare.jsonl"))
    a = ap.parse_args()
    if a.smoke:
        a.per, a.ncorrect, a.ksc, a.workers = 2, 4, 3, 3

    from datasets import load_dataset
    ds = load_dataset("hitsmy/PRMBench_Preview", split="train")
    by_cls = defaultdict(list)
    for i, c in enumerate(ds["classification"]):
        by_cls[c].append(i)
    random.seed(0)
    picks = []
    for c, idxs in by_cls.items():
        random.shuffle(idxs)
        n = a.ncorrect if c == "multi_solutions" else a.per
        picks += idxs[:n]

    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(i):
        ex = ds[i]; q = ex["question"]; steps = ex["modified_process"]; errs = ex["error_steps"] or []
        is_cor = len(errs) == 0
        p_seq = find_first(q, steps, 0.0)
        p_sc = sc_vote(q, steps, a.ksc)
        p_ps = perstep(q, steps, a.cap)
        row = {"idx": ex["idx"], "classification": ex["classification"], "error_steps": errs,
               "n_steps": len(steps), "correct_sample": is_cor,
               "seq1": {"pred": p_seq, "hit": hit_of(p_seq, errs, is_cor)},
               "sc_vote": {"pred": p_sc, "hit": hit_of(p_sc, errs, is_cor)},
               "perstep": {"pred": p_ps, "hit": hit_of(p_ps, errs, is_cor)}}
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
    lon = [r for r in err if r["n_steps"] >= 15]; sh = [r for r in err if r["n_steps"] <= 8]
    print(f"\n===== 三机制对比 (base=gemini-flash) 错误N={len(err)} 全对N={len(cor)} "
          f"长链N={len(lon)} 短链N={len(sh)} =====")
    print(f"{'机制':<20}{'recall':>8}{'误报率':>8}{'短链miss':>10}{'长链miss':>10}")
    for key, name in [("seq1", "朴素顺序(弱base)"), ("sc_vote", "顺序+SC投票(强base)"), ("perstep", "每步独立(ours)")]:
        rec = sum(r[key]["hit"] for r in err) / max(len(err), 1)
        fp = sum(1 for r in cor if not r[key]["hit"]) / max(len(cor), 1)
        ml = 1 - sum(r[key]["hit"] for r in lon) / max(len(lon), 1)
        ms = 1 - sum(r[key]["hit"] for r in sh) / max(len(sh), 1)
        print(f"{name:<20}{rec:>8.3f}{fp:>8.3f}{ms:>10.2f}{ml:>10.2f}")
    print("判据: ours 要在 recall/长链miss 上打过【顺序+SC投票】且 FP 不失控,才算真机制创新。")

if __name__ == "__main__":
    main()
