"""
对称性验证 普适性测试(PRMBench 8 类错误)。单模型。
每步在【原题 + K 个语义保持改写】下各验一次 = sym 信号(对称破缺=更多版本判错);
对照【原题重采样 K+1 次】= naive 信号。
逐类别 AUROC(错误命中率 -> 该步是否真错)。sym>naive 跨多数类别 = 普适成立。
用法: python prmbench_symmetry.py [--per 10 --k 4 --workers 10]
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

PARA = """Rewrite the following math problem {k} times, each as a DIFFERENT, aggressively DISAMBIGUATED version. In EVERY rewrite you MUST:
(1) keep all numbers and entity names EXACTLY the same;
(2) convert every percentage/fraction into an explicit multiplier (e.g. '75% of X' -> '0.75 times X');
(3) rewrite every comparative relation as explicit arithmetic (e.g. 'three fewer than Y' -> 'Y minus 3'; 'three times as many as Z' -> '3 multiplied by Z');
(4) make every referenced quantity unambiguous (state exactly which entity/time each number refers to);
(5) vary sentence structure and clause order.
Preserve the exact meaning. Output each rewrite on its own line prefixed 'R1:', 'R2:', ... .

Problem: {q}"""

VERIFY = """Problem:
{q}

Previous steps (assume correct context):
{prev}

STEP TO CHECK:
{step}

Is this step a correct inference/computation? Answer with exactly one word: YES or NO."""

def ask(prompt, temp, mx=700):
    r = client.chat.completions.create(model=MODEL, temperature=temp, max_tokens=mx,
                                       messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""

def is_err(q, prev, step, temp):
    out = ask(VERIFY.format(q=q, prev=prev, step=step), temp).strip().upper()
    m = re.search(r"\b(YES|NO)\b", out)
    return 1 if (m and m.group(1) == "NO") else 0   # NO=该步有错

def paraphrases(q, k):
    out = ask(PARA.format(k=k, q=q), 0.9, 1200)
    ps = re.findall(r"R\d+:\s*(.+)", out)
    return (ps + [q] * k)[:k]   # 不足则用原题补

def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    r = sorted(range(len(scores)), key=lambda i: scores[i])
    rank = {}; i = 0
    while i < len(r):
        j = i
        while j < len(r) and scores[r[j]] == scores[r[i]]:
            j += 1
        a = (i + 1 + j) / 2.0
        for t in range(i, j):
            rank[r[t]] = a
        i = j
    rp = sum(rank[i] for i in range(len(labels)) if labels[i] == 1)
    u = rp - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_sym.jsonl"))
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("hitsmy/PRMBench_Preview", split="train")
    by = defaultdict(list)
    for i, c in enumerate(ds["classification"]):
        if c != "multi_solutions":
            by[c].append(i)
    random.seed(0)
    picks = []
    for c, idxs in by.items():
        random.shuffle(idxs)
        picks += idxs[: a.per]

    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(ix):
        ex = ds[ix]; q = ex["question"]; steps = ex["modified_process"]; errs = ex["error_steps"] or []
        if not errs:
            return None
        e1 = min(errs)                      # 1-indexed 首错步
        if e1 < 2 or e1 > len(steps):
            return None
        cs = e1 - 1                         # 前一步=正确步
        paras = paraphrases(q, a.k)
        rows_out = []
        for (si, lab) in [(e1, 1), (cs, 0)]:   # 错误步 / 正确步
            step = steps[si - 1]
            prev = "\n".join(f"Step {j+1}: {t}" for j, t in enumerate(steps[:si-1])) or "(none)"
            # sym: 原题+K改写 各 temp0 验一次 -> 错误命中率
            sym = [is_err(q, prev, step, 0.0)] + [is_err(p, prev, step, 0.0) for p in paras]
            # naive: 原题重采样 K+1 次
            nai = [is_err(q, prev, step, 0.7) for _ in range(a.k + 1)]
            rows_out.append({"cls": ex["classification"], "label": lab,
                             "sym_rate": sum(sym) / len(sym), "naive_rate": sum(nai) / len(nai)})
        with lock:
            done[0] += 1
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if done[0] % 10 == 0:
                print(f"  ...{done[0]}", flush=True)
        return rows_out

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex_:
        futs = [ex_.submit(work, ix) for ix in picks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    rows += r
            except Exception as e:
                print(f"[skip] {e}", flush=True)
    f.close()
    report(rows)

def report(rows):
    print(f"\n===== 对称性验证 普适性 (PRMBench, N步={len(rows)}) =====", flush=True)
    print(f"{'错误类别':<22}{'AUROC-sym':>11}{'AUROC-naive':>13}{'Δ':>8}", flush=True)
    bycls = defaultdict(list)
    for r in rows:
        bycls[r["cls"]].append(r)
    wins = 0
    for c in sorted(bycls):
        g = bycls[c]
        labs = [r["label"] for r in g]
        asym = auroc([r["sym_rate"] for r in g], labs)
        anai = auroc([r["naive_rate"] for r in g], labs)
        d = asym - anai
        wins += (d > 0.02)
        print(f"{c:<22}{asym:>11.2f}{anai:>13.2f}{d:>+8.2f}{'  ✓' if d > 0.02 else ''}", flush=True)
    labs = [r["label"] for r in rows]
    print(f"{'--总计--':<22}{auroc([r['sym_rate'] for r in rows], labs):>11.2f}"
          f"{auroc([r['naive_rate'] for r in rows], labs):>13.2f}", flush=True)
    print(f"sym 更优类别数: {wins}/{len(bycls)}  -> {'普适成立' if wins > len(bycls)/2 else '不普适'}", flush=True)

if __name__ == "__main__":
    main()
