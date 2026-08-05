"""
Verify-by-Measurement 迭代4(vLLM 快版):think-then-measure。
vLLM 批量生成推理(快),再对 (prefix+reason[:K]+cue) 只读**输出首 token 的 top-20 logprobs** 取 Yes/No。
避开 prompt_logprobs 的 OOM。扫 K,直接对比 GenPRM-1.5B。
用法: python measure_think_vllm.py --model <path> --n_err 20 --n_cor 20 --genprm out_math10.jsonl
"""
import os, json, argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PB_DIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SUBSETS = ["math"]
SYS = "You are a math teacher. Analyze whether the specified step is correct."
KS = [128, 256, 512, 768]
CUE = "\n\nFinal judgment — is this step correct? Answer with only Yes or No.\nAnswer:"


def load_cases(n_err, n_cor):
    err, cor = [], []
    for sub in SUBSETS:
        p = os.path.join(PB_DIR, f"{sub}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            lab = int(r["label"]); steps = r["steps"]
            if isinstance(steps, list) and len(steps) >= 2:
                (cor if lab == -1 else err).append((r["id"], r["problem"], steps, lab))
    return err[:n_err], cor[:n_cor]


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    w = t = 0
    for p in pos:
        for n in neg:
            if p > n: w += 1
            elif p == n: t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


def tok_ids(tok, words):
    s = set()
    for w in words:
        for v in (w, " " + w):
            e = tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return list(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_err", type=int, default=20)
    ap.add_argument("--n_cor", type=int, default=20)
    ap.add_argument("--genprm", default=None)
    ap.add_argument("--out", default="results_think_vllm.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    yes_ids = set(tok_ids(tok, ["Yes", "yes", "Correct", "correct"]))
    no_ids = set(tok_ids(tok, ["No", "no", "Incorrect", "incorrect", "Wrong", "wrong"]))
    cue_ids = tok(CUE, add_special_tokens=False).input_ids

    err, cor = load_cases(a.n_err, a.n_cor)
    cases = err + cor
    print(f"cases: err={len(err)} cor={len(cor)}", flush=True)

    # 构造每步 base 提示
    meta, base_list = [], []
    for (cid, q, steps, lab) in cases:
        n = len(steps)
        sol = "\n".join(f"Step {j+1}: {steps[j]}" for j in range(n))
        for t in range(n):
            if lab >= 0 and t > lab:
                continue
            user = f"Question: {q}\n\nSolution:\n{sol}\n\nAnalyze whether Step {t+1} is correct."
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if len(ids) > 3000:
                continue
            meta.append(dict(cid=cid, t=t, is_err=int(lab >= 0 and t == lab), chain_len=n))
            base_list.append(ids)
    print(f"steps={len(meta)} pos={sum(m['is_err'] for m in meta)}", flush=True)

    llm = LLM(model=a.model, dtype="half", max_model_len=5000,
              gpu_memory_utilization=0.6, enforce_eager=True)

    # 阶段1:批量生成推理
    gen_sp = SamplingParams(temperature=0.0, max_tokens=max(KS))
    gout = llm.generate([{"prompt_token_ids": b} for b in base_list], gen_sp)
    reasons = [o.outputs[0].token_ids for o in gout]

    # 阶段2:每个 K 读 Yes/No(输出首 token top-20 logprobs)
    vsp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    for m in meta:
        m["s"] = {}
    for K in KS:
        prompts = [{"prompt_token_ids": base_list[i] + list(reasons[i][:K]) + cue_ids}
                   for i in range(len(meta))]
        vout = llm.generate(prompts, vsp)
        for i, o in enumerate(vout):
            lp = o.outputs[0].logprobs[0]  # dict {tid: Logprob} top-20 at pos0
            import math
            py = sum(math.exp(v.logprob) for tid, v in lp.items() if tid in yes_ids)
            pn = sum(math.exp(v.logprob) for tid, v in lp.items() if tid in no_ids)
            meta[i]["s"][K] = math.log(py + 1e-12) - math.log(pn + 1e-12)
        labs = [m["is_err"] for m in meta]
        au = auroc([-m["s"][K] for m in meta], labs)
        print(f"  K={K:>3}  AUROC={au:.3f}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    labs = [m["is_err"] for m in meta]
    print(f"\n===== think-then-measure vLLM (N={len(meta)}, 错步={sum(labs)}) =====", flush=True)
    for K in KS:
        print(f"  K={K:>3}  AUROC={auroc([-m['s'][K] for m in meta], labs):.3f}", flush=True)
    if a.genprm and os.path.exists(a.genprm):
        gp = {json.loads(l)['id']: json.loads(l) for l in open(a.genprm, encoding='utf-8') if l.strip()}
        gel, gl = [], []
        for m in meta:
            g = gp.get(m['cid'])
            if g and m['t'] < len(g['scores']):
                gel.append(1 - g['scores'][m['t']]); gl.append(m['is_err'])
        if gl:
            print(f"  [重叠{len(gl)}步] GenPRM-1.5B AUROC={auroc(gel, gl):.3f}", flush=True)


if __name__ == "__main__":
    main()
