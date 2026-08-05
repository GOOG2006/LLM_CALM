"""
同 1.5B 基座(DeepSeek-R1-Distill-Qwen-1.5B)跑两机制,与 GenPRM-1.5B 比:
  corr: 每步 read-and-judge(相关式)
  ind : 每步独立重算 K 次 + 共识闸(去相关)
vLLM 批量生成。远程运行(读 genprm_preds.json)。
用法: python verify_1p5b.py --k 5
"""
import os, re, json, argparse
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "/root/autodl-tmp/genprm_work/models/DSR1-1.5B"
PREDS = "/root/autodl-tmp/genprm_work/genprm_preds.json"

CORR = """Problem:
{q}

Previous steps:
{prev}

STEP TO CHECK:
Step {i}: {step}

Is this step a valid, correct inference/computation? Reason briefly, then end with exactly 'Verdict: CORRECT' or 'Verdict: ERROR'."""

RECOMP = """Problem:
{q}

Work so far (previous steps, assume correct):
{prev}

The next step written by a student is: {step}

Independently compute the final numerical value this step SHOULD arrive at, using ONLY the problem and previous steps. Do NOT reuse the arithmetic inside the student step. End your answer with 'ANSWER: <number>'."""

def last_num(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", (t or "").replace(",", ""))
    try:
        return float(m[-1]) if m else None
    except ValueError:
        return None

def ans_num(t):
    m = re.search(r"ANSWER:\s*(-?\d[\d,]*\.?\d*)", t or "")
    return last_num(m.group(1)) if m else last_num(t)

def eq(a, b):
    if a is None or b is None:
        return True
    return abs(a - b) <= 1e-6 + 1e-3 * abs(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    data = json.load(open(PREDS, encoding="utf-8"))[: a.n]
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="half", enable_chunked_prefill=False, enforce_eager=False,
              tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.9)

    def chat(msg):
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       add_generation_prompt=True, tokenize=False)

    # 收集所有 prompt
    corr_prompts, corr_idx = [], []   # (sample, step)
    rec_prompts, rec_idx = [], []
    for si, d in enumerate(data):
        q, steps = d["problem"], d["steps"]
        for i, s in enumerate(steps):
            prev = "\n".join(f"Step {j+1}: {t}" for j, t in enumerate(steps[:i])) or "(none)"
            corr_prompts.append(chat(CORR.format(q=q, prev=prev, i=i + 1, step=s))); corr_idx.append((si, i))
            rec_prompts.append(chat(RECOMP.format(q=q, prev=prev, step=s))); rec_idx.append((si, i))

    sp_corr = SamplingParams(temperature=0.0, max_tokens=700)
    sp_rec = SamplingParams(temperature=0.8, max_tokens=700, n=a.k)
    print(f"generating: corr={len(corr_prompts)} rec={len(rec_prompts)}x{a.k}", flush=True)
    corr_out = llm.generate(corr_prompts, sp_corr)
    rec_out = llm.generate(rec_prompts, sp_rec)

    # 组织每样本每步结果
    S = [{"steps": len(d["steps"]), "corr": {}, "rec": {}} for d in data]
    for (si, i), o in zip(corr_idx, corr_out):
        txt = o.outputs[0].text
        m = re.search(r"Verdict:\s*(CORRECT|ERROR)", txt, re.I)
        S[si]["corr"][i] = (m and m.group(1).upper() == "ERROR")
    for (si, i), o in zip(rec_idx, rec_out):
        vals = [ans_num(x.text) for x in o.outputs]
        vals = [v for v in vals if v is not None]
        gi = last_num(data[si]["steps"][i])
        if not vals or gi is None:
            S[si]["rec"][i] = (0.0, False)
        else:
            cnt = Counter([round(v, 4) for v in vals]); mode, freq = cnt.most_common(1)[0]
            S[si]["rec"][i] = (freq / len(vals), not eq(mode, gi))

    rows = []
    for si, d in enumerate(data):
        corr_pred = next((i for i in range(S[si]["steps"]) if S[si]["corr"].get(i)), -1)
        rows.append({"label": d["label"], "genprm_pred": d["genprm_pred"],
                     "corr_pred": corr_pred, "rec": [S[si]["rec"].get(i, (0.0, False)) for i in range(S[si]["steps"])]})
    json.dump(rows, open("/root/autodl-tmp/genprm_work/verify_1p5b_out.json", "w"))
    report(rows)

def ind_at(rec, tau):
    for i, (c, mm) in enumerate(rec):
        if mm and c >= tau:
            return i
    return -1

def report(rows):
    err = [r for r in rows if r["label"] != -1]; cor = [r for r in rows if r["label"] == -1]
    def st(fn, name):
        ae = sum(fn(r) == r["label"] for r in err) / max(len(err), 1)
        ac = sum(fn(r) == -1 for r in cor) / max(len(cor), 1)
        f1 = 2 * ae * ac / max(ae + ac, 1e-9)
        print(f"{name:<30} error-acc={ae:.3f} correct-acc={ac:.3f} F1={f1*100:.1f}", flush=True)
    print(f"\n===== 全 1.5B 基座 (错{len(err)}/对{len(cor)}) =====", flush=True)
    st(lambda r: r["genprm_pred"], "GenPRM-1.5B(专用,官方)")
    st(lambda r: r["corr_pred"], "base-1.5B 相关式read-judge")
    for tau in [0.0, 0.6, 0.8, 1.0]:
        st(lambda r, t=tau: ind_at(r["rec"], t), f"base-1.5B 独立+闸 tau={tau}")

if __name__ == "__main__":
    main()
