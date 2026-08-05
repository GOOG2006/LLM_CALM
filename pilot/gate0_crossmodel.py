"""
Gate 0(方向 C)存在性测 — 见 PREREG_C_gate0.md。
问题:在 base 自洽子集里,跨模型分歧能否分开 base 的对/错?
不发无谓调用;每题存真实答案,便于日后并入 GPT checker。
用法:
  python gate0_crossmodel.py --smoke
  python gate0_crossmodel.py --n 500 --kbase 6 --kcheck 4 --workers 8
环境:GEMINI_API_KEY 必需。BASE_MODEL / CHECK_MODEL 可覆盖。
"""
import os, re, json, argparse, sys, random, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from openai import OpenAI

client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
BASE_MODEL = os.environ.get("BASE_MODEL", "gemini-2.5-flash-lite")
CHECK_MODEL = os.environ.get("CHECK_MODEL", "gemini-2.5-flash")
SYS = "Solve the math problem step by step. End with a line 'Answer: <number>'."

def extract_num(text):
    if text is None:
        return None
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None

def gold_of(s):
    return extract_num(s["answer"].split("####")[-1])

def is_correct(p, g):
    return p is not None and g is not None and abs(p - g) < 1e-4

def ans(model, q, temperature):
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=1024,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": q}])
    t = r.choices[0].message.content
    return extract_num(t.split("Answer:")[-1] if "Answer:" in t else t)

def majority(xs):
    v = [round(x, 4) for x in xs if x is not None]
    if not v:
        return None, 1.0
    top, cnt = Counter(v).most_common(1)[0]
    return top, 1 - cnt / len(v)   # (多数答案, 分歧度)

def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])
    rank, i = {}, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        a = (i + 1 + j) / 2.0
        for k in range(i, j):
            rank[ranked[k][1]] = a
        i = j
    rpos = sum(rank[idx] for idx, y in enumerate(labels) if y == 1)
    u = rpos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--kbase", type=int, default=6)
    ap.add_argument("--kcheck", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_gate0.jsonl"))
    a = ap.parse_args()
    if a.smoke:
        a.n, a.kbase, a.kcheck, a.workers = 3, 3, 2, 3

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(0)
    idxs = random.sample(range(len(ds)), a.n)

    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(ix):
        q, gold = ds[ix]["question"], gold_of(ds[ix])
        b = [ans(BASE_MODEL, q, 0.8) for _ in range(a.kbase)]
        base_ans, base_dis = majority(b)
        c = [ans(CHECK_MODEL, q, 0.8) for _ in range(a.kcheck)]
        cv = [x for x in c if x is not None]
        cross = 1.0 if not cv else sum(1 for x in cv if not is_correct(x, base_ans)) / len(cv)
        row = {"id": ix, "gold": gold, "base_ans": base_ans,
               "base_disagree": round(base_dis, 4), "correct": int(is_correct(base_ans, gold)),
               "cross_disagree": round(cross, 4)}
        with lock:
            done[0] += 1
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"[{done[0]}/{a.n}] id={ix} correct={row['correct']} "
                  f"base_dis={row['base_disagree']:.2f} cross_dis={row['cross_disagree']:.2f}", flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, ix): ix for ix in idxs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[skip] id={futs[fut]} {e}", flush=True)
    f.close()
    report(rows)

def report(rows):
    labels = [1 - r["correct"] for r in rows]   # 正类=base 答错
    def blk(name, sub):
        lab = [1 - r["correct"] for r in sub]
        nw = sum(lab)
        auc_cross = auroc([r["cross_disagree"] for r in sub], lab)
        auc_self = auroc([r["base_disagree"] for r in sub], lab)
        print(f"{name}: N={len(sub)} 错={nw}  "
              f"AUROC[跨模型]={auc_cross:.3f}  AUROC[self-cons]={auc_self:.3f}")
        return auc_cross, nw
    print("\n==================== Gate 0 判决 ====================")
    print(f"全集 N={len(rows)}  总错={sum(labels)}")
    blk("全集      ", rows)
    strict = [r for r in rows if r["base_disagree"] == 0.0]
    near = [r for r in rows if r["base_disagree"] <= 0.2]
    auc_s, nw_s = blk("严格自洽  ", strict)   # ← 主判决
    blk("近自洽≤.2 ", near)
    print("-" * 52)
    if nw_s < 15:
        print(f"裁决: 灰区 — 严格自洽子集错答仅 {nw_s}<15,欠功效,需加 N / 明天并 GPT checker")
    elif auc_s >= 0.65:
        print(f"裁决: Gate 0 过(AUROC={auc_s:.3f}) — 正交信号存在 → 方向 C 活;Gate 1 须换异家族 checker 排除能力混淆")
    elif auc_s < 0.55:
        print(f"裁决: 判死(AUROC={auc_s:.3f}) — 跨模型也抓不住自洽-错,墙确认")
    else:
        print(f"裁决: 灰区(AUROC={auc_s:.3f}) — 加 N 或换异家族 checker 复测")

if __name__ == "__main__":
    main()
