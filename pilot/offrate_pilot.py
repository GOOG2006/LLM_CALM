"""
存在性测量 — 种子 B(动力学校对)。见 PILOT_PREREG.md。
唯一目标:测两条 AUROC —— (1) 中点重推导 off_rate,(2) 全新生成答案分歧(对照闸①)——
作为"该题答对"的预测子。AUROC≥0.65 过闸;<0.55 判死;中间灰区换 MATH。

用法:
  python offrate_pilot.py --smoke          # N=2,K=3,验证管路
  python offrate_pilot.py --n 60 --k 8     # 正式跑
环境: OPENAI_API_KEY 必需;PILOT_MODEL 可选(默认 gpt-4o-mini)。
"""
import os, re, json, argparse, math, random, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK,强制 UTF-8
except Exception:
    pass
from openai import OpenAI

BACKEND = os.environ.get("PILOT_BACKEND", "openai")
if BACKEND == "gemini":
    client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    MODEL = os.environ.get("PILOT_MODEL", "gemini-2.5-flash")
else:
    client = OpenAI()
    MODEL = os.environ.get("PILOT_MODEL", "gpt-4o-mini")

# ---------- 答案抽取 / 判分 ----------
def extract_num(text):
    """取文本中最后一个数字(去千分位),GSM8K 答案均为整数/有限小数。"""
    if text is None:
        return None
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None

def gold_of(sample):
    return extract_num(sample["answer"].split("####")[-1])

def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < 1e-4

# ---------- 模型调用 ----------
def gen(messages, temperature, max_tokens=512):
    r = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return r.choices[0].message.content

SYS = "Solve the math problem step by step. End with a line 'Answer: <number>'."

def reference_solution(q):
    txt = gen([{"role": "system", "content": SYS},
               {"role": "user", "content": q}], temperature=0.7)
    return txt, extract_num(txt.split("Answer:")[-1] if "Answer:" in txt else txt)

def midpoint_prefix(solution):
    """把参考链在中点截断,作为'检查点'。按行,不足则按句。"""
    lines = [l for l in solution.splitlines() if l.strip()]
    if len(lines) >= 4:
        return "\n".join(lines[: max(1, len(lines) // 2)])
    sents = re.split(r"(?<=[.。])\s+", solution)
    return " ".join(sents[: max(1, len(sents) // 2)])

def rederive(q, prefix, k, temperature=0.9):
    """从检查点重采样续写 k 次,返回 k 个最终答案。"""
    ans = []
    for _ in range(k):
        txt = gen([{"role": "system", "content": SYS},
                   {"role": "user", "content": q},
                   {"role": "assistant", "content": prefix},
                   {"role": "user", "content": "Continue from where the solution stops and finish with 'Answer: <number>'."}],
                  temperature=temperature)
        ans.append(extract_num(txt.split("Answer:")[-1] if "Answer:" in txt else txt))
    return ans

def fresh(q, k, temperature=0.9):
    """对照闸①:k 次全新完整生成(无截断)。"""
    return [reference_solution(q)[1] for _ in range(k)]

# ---------- 指标 ----------
def off_rate(ref_ans, samples):
    valid = [a for a in samples if a is not None]
    if not valid:
        return 1.0
    return sum(1 for a in valid if not is_correct(a, ref_ans)) / len(valid)

def disagreement(samples):
    """全新生成的分歧率:1 - 最大众数占比(self-consistency 的不确定性)。"""
    valid = [round(a, 4) for a in samples if a is not None]
    if not valid:
        return 1.0
    from collections import Counter
    top = Counter(valid).most_common(1)[0][1]
    return 1 - top / len(valid)

def auroc(scores, labels):
    """labels: 1=答错(正类,期望 score 高). Mann-Whitney U,手写。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])
    rank = {}
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0  # 1-based 平均秩,处理 ties
        for k in range(i, j):
            rank[ranked[k][1]] = avg
        i = j
    rpos = sum(rank[idx] for idx, y in enumerate(labels) if y == 1)
    u = rpos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)  # 并发题数
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.jsonl"))
    args = ap.parse_args()
    if args.smoke:
        args.n, args.k = 2, 3

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(0)
    idxs = random.sample(range(len(ds)), args.n)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()
    f = open(args.out, "w", encoding="utf-8")
    done = [0]

    def work(ix):
        """单题全流程(1 参考 + k 重推导 + k 全新)。线程安全:仅写文件/计数时加锁。"""
        q, gold = ds[ix]["question"], gold_of(ds[ix])
        sol, ref_ans = reference_solution(q)
        correct = is_correct(ref_ans, gold)
        prefix = midpoint_prefix(sol)
        red = rederive(q, prefix, args.k)
        frn = fresh(q, args.k)
        row = {
            "id": ix,
            "ref_correct": int(correct),
            "off_rate_rederive": off_rate(ref_ans, red),   # 相对参考答案的脱落率
            "disagree_fresh": disagreement(frn),            # 对照:全新生成分歧
            "chain_len": len([l for l in sol.splitlines() if l.strip()]),
        }
        with lock:
            done[0] += 1
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"[{done[0]}/{args.n}] id={ix} correct={row['ref_correct']} "
                  f"off_rate={row['off_rate_rederive']:.2f} disagree={row['disagree_fresh']:.2f}",
                  flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, ix): ix for ix in idxs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[skip] id={futs[fut]} error: {e}", flush=True)
    f.close()

    labels = [1 - r["ref_correct"] for r in rows]  # 正类=答错
    n_wrong, n_right = sum(labels), len(labels) - sum(labels)
    auc_red = auroc([r["off_rate_rederive"] for r in rows], labels)
    auc_fresh = auroc([r["disagree_fresh"] for r in rows], labels)

    print("\n==================== 判死/过闸 ====================")
    print(f"N={len(rows)}  答错={n_wrong}  答对={n_right}  (两桶各需≥15 才算有效)")
    print(f"AUROC[中点重推导 off_rate]  = {auc_red:.3f}   ← 主指标")
    print(f"AUROC[全新生成 disagree]    = {auc_fresh:.3f}   ← 对照(self-consistency)")
    lift = auc_red - auc_fresh
    print(f"闸①增益 (重推导 - 对照)     = {lift:+.3f}   (>~0.03 才算'校对'有独立价值)")
    verdict = ("过闸→进 Gate2" if auc_red >= 0.65 else
               "判死→写 FINDINGS" if auc_red < 0.55 else "灰区→换 MATH 复测")
    print(f"主指标裁决: {verdict}")
    if auc_red >= 0.65 and lift < 0.03:
        print("⚠ 注意: 主指标过闸但闸①增益不足 → 疑似 self-consistency 换皮,需 MATH 上复核增益。")

if __name__ == "__main__":
    main()
