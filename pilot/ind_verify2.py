"""
v2: 独立重导+共识闸 + 同 base 相关式对照。对同一批 ProcessBench 样本。
每步:①独立重算 K 次 -> 记 (consensus, mode!=gi);②相关式 read-and-judge 一次。
共识闸: 只有 consensus>=tau 且 mode!=gi 才 flag(离线可扫 tau)。
用法: python ind_verify2.py [--k 5 --workers 10 --n 100]
"""
import os, re, json, argparse, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.environ.get("PILOT_MODEL", "gemini-2.5-flash")

RECOMPUTE = """Problem:
{q}

Work so far (previous steps, assume correct context):
{prev}

The NEXT step written by a student is: {step}

Independently compute the final numerical value this step SHOULD arrive at, using ONLY the problem and previous steps. Do NOT reuse or trust the arithmetic/substitutions inside the student step. Output ONLY the final number."""

CORR = """Problem:
{q}

Previous steps:
{prev}

STEP TO CHECK:
Step {i}: {step}

Is this step a valid, correct inference/computation? Reason in 1-2 sentences, then end with exactly 'Verdict: CORRECT' or 'Verdict: ERROR'."""

def last_num(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", (t or "").replace(",", ""))
    try:
        return float(m[-1]) if m else None
    except ValueError:
        return None

def eq(a, b):
    if a is None or b is None:
        return True
    return abs(a - b) <= 1e-6 + 1e-3 * abs(b)

def ask(prompt, temp, mx=512):
    r = client.chat.completions.create(model=MODEL, temperature=temp, max_tokens=mx,
                                       messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""

def analyze_sample(d, k):
    q, steps = d["problem"], d["steps"]
    per_step = []      # 每步 {c, mm}  给独立+共识闸
    corr_pred = -1
    for i, s in enumerate(steps):
        prev = "\n".join(f"Step {j+1}: {t}" for j, t in enumerate(steps[:i])) or "(none)"
        # 相关式判(同 base)
        if corr_pred == -1:
            out = ask(CORR.format(q=q, prev=prev, i=i + 1, step=s), 0.0)
            m = re.search(r"Verdict:\s*(CORRECT|ERROR)", out, re.I)
            if m and m.group(1).upper() == "ERROR":
                corr_pred = i
        # 独立重算 K 次
        gi = last_num(s)
        if gi is None:
            per_step.append({"c": 0.0, "mm": False}); continue
        vals = [last_num(ask(RECOMPUTE.format(q=q, prev=prev, step=s), 0.8)) for _ in range(k)]
        vals = [v for v in vals if v is not None]
        if not vals:
            per_step.append({"c": 0.0, "mm": False}); continue
        cnt = Counter([round(v, 4) for v in vals]); mode, freq = cnt.most_common(1)[0]
        per_step.append({"c": freq / len(vals), "mm": not eq(mode, gi)})
    return corr_pred, per_step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_ind2.jsonl"))
    a = ap.parse_args()
    data = json.load(open(os.path.join(os.path.dirname(__file__), "genprm_preds.json"), encoding="utf-8"))[: a.n]
    lock = threading.Lock(); done = [0]
    f = open(a.out, "w", encoding="utf-8")

    def work(d):
        corr_pred, per_step = analyze_sample(d, a.k)
        row = {"name": d["name"], "label": d["label"], "genprm_pred": d["genprm_pred"],
               "corr_pred": corr_pred, "per_step": per_step}
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

def ind_pred_at(per_step, tau):
    for i, ps in enumerate(per_step):
        if ps["mm"] and ps["c"] >= tau:
            return i
    return -1

def report(rows):
    err = [r for r in rows if r["label"] != -1]; cor = [r for r in rows if r["label"] == -1]
    def stats(predfn, name):
        ae = sum(predfn(r) == r["label"] for r in err) / max(len(err), 1)
        ac = sum(predfn(r) == -1 for r in cor) / max(len(cor), 1)
        f1 = 2 * ae * ac / max(ae + ac, 1e-9)
        print(f"{name:<28} error-acc={ae:.3f}  correct-acc={ac:.3f}  F1={f1*100:.1f}")
    print(f"\n===== v2 (同 {len(rows)} 样本, 错{len(err)}/对{len(cor)}) =====")
    stats(lambda r: r["genprm_pred"], "GenPRM(1.5B官方)")
    stats(lambda r: r["corr_pred"], "相关式 read-judge(同base gemini)")
    for tau in [0.0, 0.6, 0.8, 1.0]:
        stats(lambda r, t=tau: ind_pred_at(r["per_step"], t), f"独立+共识闸 tau={tau}")

if __name__ == "__main__":
    main()
