"""
对抗式立场去相关(单模型)。同输入、同K、只换"立场"。对照 K 次中性重采样。
每步存 4 个立场判定 + 4 个中性判定 -> AUROC + proper互补性 + 一致性类 + 过度flag。
用法: python prmbench_stance.py [--per 10 --workers 10]
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

HEAD = "Problem:\n{q}\n\nPrevious steps (assume correct context):\n{prev}\n\nSTEP TO CHECK:\n{step}\n\n"
TAIL = " Answer with exactly one word: YES (the step is correct) or NO (the step contains an error)."
# 4 个对立立场
STANCES = [
    "You are a prosecutor. Assume this step CONTAINS an error and try hard to identify the specific mistake. Only if you genuinely cannot find any error, conclude it is correct.",
    "Do NOT trust this step. Independently compute what this step should produce using only the problem and previous steps, then compare to what the step claims.",
    "Check specifically whether this step uses the correct quantity for the correct entity/time and interprets the problem's wording correctly.",
    "Carefully judge whether this step is a valid, correct inference.",   # 中性(=naive 那一种)
]
NEUTRAL = STANCES[3]

def verdict(q, prev, step, stance, temp):
    out = client.chat.completions.create(model=MODEL, temperature=temp, max_tokens=600,
        messages=[{"role": "user", "content": HEAD.format(q=q, prev=prev, step=step) + stance + TAIL}]
        ).choices[0].message.content or ""
    m = re.search(r"\b(YES|NO)\b", out.strip().upper())
    return 1 if (m and m.group(1) == "NO") else 0   # 1 = 判该步有错

def auroc(sc, la):
    pos = [s for s, y in zip(sc, la) if y == 1]; neg = [s for s, y in zip(sc, la) if y == 0]
    if not pos or not neg:
        return float("nan")
    r = sorted(range(len(sc)), key=lambda i: sc[i]); rk = {}; i = 0
    while i < len(r):
        j = i
        while j < len(r) and sc[r[j]] == sc[r[i]]:
            j += 1
        a = (i + 1 + j) / 2
        for t in range(i, j):
            rk[r[t]] = a
        i = j
    rp = sum(rk[i] for i in range(len(la)) if la[i] == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=10); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_stance.jsonl"))
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("hitsmy/PRMBench_Preview", split="train")
    by = defaultdict(list)
    for i, c in enumerate(ds["classification"]):
        if c != "multi_solutions":
            by[c].append(i)
    random.seed(0); picks = []
    for c, idxs in by.items():
        random.shuffle(idxs); picks += idxs[: a.per]
    lock = threading.Lock(); done = [0]; f = open(a.out, "w", encoding="utf-8")

    def work(ix):
        ex = ds[ix]; q = ex["question"]; steps = ex["modified_process"]; errs = ex["error_steps"] or []
        if not errs:
            return None
        e1 = min(errs)
        if e1 < 2 or e1 > len(steps):
            return None
        out = []
        for si, lab in [(e1, 1), (e1 - 1, 0)]:
            step = steps[si - 1]
            prev = "\n".join(f"Step {j+1}: {t}" for j, t in enumerate(steps[:si-1])) or "(none)"
            stance = [verdict(q, prev, step, s, 0.0) for s in STANCES]     # 4 立场
            naive = [verdict(q, prev, step, NEUTRAL, 0.7) for _ in range(4)]  # 4 中性重采样
            out.append({"cls": ex["classification"], "label": lab, "stance": stance, "naive": naive})
        with lock:
            done[0] += 1
            for r in out:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if done[0] % 10 == 0:
                print(f"  ...{done[0]}", flush=True)
        return out

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
    f.close(); report(rows)

def report(rows):
    err = [r for r in rows if r["label"] == 1]; cor = [r for r in rows if r["label"] == 0]
    print(f"\n===== 对抗式立场去相关 (PRMBench, 错步{len(err)}/对步{len(cor)}) =====", flush=True)
    # 1) 逐类别 AUROC
    print(f"{'类别':<20}{'AUROC-stance':>13}{'AUROC-naive':>13}{'Δ':>7}", flush=True)
    by = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    wins = 0
    for c in sorted(by):
        g = by[c]; la = [r["label"] for r in g]
        As = auroc([sum(r["stance"]) / 4 for r in g], la); An = auroc([sum(r["naive"]) / 4 for r in g], la)
        d = As - An; wins += (d > 0.02)
        print(f"{c:<20}{As:>13.2f}{An:>13.2f}{d:>+7.2f}{'  ✓' if d > 0.02 else ''}", flush=True)
    la = [r["label"] for r in rows]
    print(f"{'总计':<20}{auroc([sum(r['stance'])/4 for r in rows], la):>13.2f}"
          f"{auroc([sum(r['naive'])/4 for r in rows], la):>13.2f}", flush=True)
    print(f"stance 更优类别: {wins}/{len(by)}", flush=True)
    # 2) proper 互补性(金标错误步)
    def comp(key):
        anyc = sum(1 for r in err if any(r[key])) / max(len(err), 1)
        allc = sum(1 for r in err if all(r[key])) / max(len(err), 1)
        return anyc, allc
    sa, sall = comp("stance"); na, nall = comp("naive")
    print(f"\nproper互补(错步): stance 任一抓={sa:.2f} 都抓={sall:.2f} 差={sa-sall:.2f} | "
          f"naive 任一={na:.2f} 都抓={nall:.2f} 差={na-nall:.2f}", flush=True)
    print(f"  -> 立场互补差 {'>' if (sa-sall)>(na-nall) else '<='} naive互补差  ({'真去相关' if (sa-sall)>(na-nall) else '没去相关'})", flush=True)
    # 3) 过度flag(对步被判错的比例)
    sf = sum(sum(r["stance"]) for r in cor) / max(4*len(cor), 1)
    nf = sum(sum(r["naive"]) for r in cor) / max(4*len(cor), 1)
    print(f"过度flag(对步误判率): stance={sf:.2f} naive={nf:.2f}", flush=True)

if __name__ == "__main__":
    main()
