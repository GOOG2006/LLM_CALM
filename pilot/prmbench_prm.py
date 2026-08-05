"""
PRMBench 上 LLM-as-PRM 残余缺陷分析 — 见 PREREG_prmbench.md。
任务:给 question+编号步骤,模型输出第一个错误步号(全对=0),与 error_steps 比。
输出每类 miss-rate + 整体 recall + 误报率。gemini-flash 后端。
用法:
  python prmbench_prm.py --smoke
  python prmbench_prm.py --per 20 --workers 8
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

PROMPT = """You are a careful math reasoning verifier. Below is a problem and a step-by-step solution with numbered steps. Exactly identify the FIRST step that contains an error (mathematical, logical, or unfounded claim). If EVERY step is correct, answer 0.

Problem:
{q}

Solution steps:
{steps}

Think briefly, then end with a line exactly: 'First error step: <N>' where N is the step number (or 0 if no error."""

def parse_step(text, n):
    if text is None:
        return None
    m = re.search(r"First error step:\s*(\d+)", text)
    if m:
        v = int(m.group(1))
        return v if 0 <= v <= n else None
    nums = re.findall(r"\b(\d+)\b", text)
    return int(nums[-1]) if nums else None

def prm_call(q, steps):
    numbered = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    r = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT.format(q=q, steps=numbered)}])
    return parse_step(r.choices[0].message.content, len(steps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20)      # 每类样本数
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_prmbench.jsonl"))
    a = ap.parse_args()
    if a.smoke:
        a.per, a.workers = 2, 3

    from datasets import load_dataset
    ds = load_dataset("hitsmy/PRMBench_Preview", split="train")

    # 分层抽样:每个 classification 取 per 条
    by_cls = defaultdict(list)
    for i, c in enumerate(ds["classification"]):
        by_cls[c].append(i)
    random.seed(0)
    picks = []
    for c, idxs in by_cls.items():
        random.shuffle(idxs)
        picks += idxs[:a.per]

    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(i):
        ex = ds[i]
        q = ex["question"]; steps = ex["modified_process"]; errs = ex["error_steps"] or []
        pred = prm_call(q, steps)
        is_correct_sample = len(errs) == 0
        if is_correct_sample:
            hit = (pred == 0)          # 全对样本:应答 0;答非 0 = 误报
        else:
            hit = (pred in errs)       # 错误样本:命中任一错误步
        row = {"idx": ex["idx"], "classification": ex["classification"],
               "error_steps": errs, "pred_step": pred, "n_steps": len(steps),
               "correct_sample": is_correct_sample, "hit": bool(hit)}
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
    print("\n==================== PRMBench LLM-as-PRM ====================")
    err = [r for r in rows if not r["correct_sample"]]
    cor = [r for r in rows if r["correct_sample"]]
    overall_recall = sum(r["hit"] for r in err) / max(len(err), 1)
    print(f"整体 error-recall = {overall_recall:.3f}  (SOTA 对齐区 ~0.5-0.7)  错误样本 N={len(err)}")
    if cor:
        fp = sum(1 for r in cor if not r["hit"]) / len(cor)
        print(f"误报率(全对样本被判错) = {fp:.3f}  N={len(cor)}")
    print("-- 每类 miss-rate(漏检率,高=可攻缺陷候选) --")
    by = defaultdict(list)
    for r in err:
        by[r["classification"]].append(r["hit"])
    for c in sorted(by, key=lambda c: -(1 - sum(by[c]) / len(by[c]))):
        miss = 1 - sum(by[c]) / len(by[c])
        flag = "  ← 缺陷候选(≥0.40)" if miss >= 0.40 else ""
        print(f"  {c:22s} miss={miss:.2f}  (N={len(by[c])}){flag}")

if __name__ == "__main__":
    main()
