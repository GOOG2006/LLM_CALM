"""
Verify-by-Measurement 迭代5:think-measure + 自洽平均(去噪)。
K=512 固定,采样 N 条推理(temp>0),平均判决 logit s。对比单条(0.783)与 GenPRM(0.848)。
用法: python measure_think_sc.py --model <path> --n_err 20 --n_cor 20 --samples 6 --K 512 --genprm out_math10.jsonl
"""
import os, json, argparse, math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PB_DIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SUBSETS = ["math"]
SYS = ("You are a rigorous math grader. To check a step, independently re-derive its result "
       "from the problem and previous steps, verify each calculation and logical inference, "
       "and watch for wrong formulas, dropped conditions, sign/arithmetic errors, and unjustified leaps.")
USER_TMPL = ("Question: {q}\n\nSolution:\n{sol}\n\n"
             "Carefully verify ONLY Step {i}. Re-derive it independently, compare with your own "
             "derivation, and pinpoint any error. Reason step by step.")
CUE = "\n\nConclusion — is Step {i} correct? Answer with only Yes or No.\nAnswer:"


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
    pos = [s for s, l in zip(scores, labels) if l == 1]; neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return float("nan")
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
    ap.add_argument("--n_err", type=int, default=20); ap.add_argument("--n_cor", type=int, default=20)
    ap.add_argument("--samples", type=int, default=6); ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--genprm", default=None); ap.add_argument("--out", default="results_think_sc.jsonl")
    ap.add_argument("--subset", default="math")
    a = ap.parse_args()
    global SUBSETS
    SUBSETS = [a.subset]

    tok = AutoTokenizer.from_pretrained(a.model)
    yes_ids = set(tok_ids(tok, ["Yes", "yes", "Correct", "correct"]))
    no_ids = set(tok_ids(tok, ["No", "no", "Incorrect", "incorrect", "Wrong", "wrong"]))

    err, cor = load_cases(a.n_err, a.n_cor)
    # 续跑:已在输出文件里的 cid 跳过
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["cid"])
    meta, base_list, cue_list = [], [], []
    for (cid, q, steps, lab) in err + cor:
        if cid in done:
            continue
        n = len(steps); sol = "\n".join(f"Step {j+1}: {steps[j]}" for j in range(n))
        for t in range(n):
            if lab >= 0 and t > lab: continue
            user = USER_TMPL.format(q=q, sol=sol, i=t + 1)
            ids = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                          add_generation_prompt=True, tokenize=True)
            if len(ids) > 11500: continue  # 与 GenPRM 的 max_model_len=12000 对齐,公平覆盖长链
            meta.append(dict(cid=cid, t=t, is_err=int(lab >= 0 and t == lab))); base_list.append(ids)
            cue_list.append(tok(CUE.format(i=t + 1), add_special_tokens=False).input_ids)
    print(f"steps={len(meta)} pos={sum(m['is_err'] for m in meta)} samples={a.samples} K={a.K} (resume: {len(done)} cids done)", flush=True)
    if not meta:
        print("nothing new to run", flush=True); return

    llm = LLM(model=a.model, dtype="half", max_model_len=13000, gpu_memory_utilization=0.7, enforce_eager=True, max_num_seqs=16)
    gen_sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.K, n=a.samples, seed=0)
    gout = llm.generate([{"prompt_token_ids": b} for b in base_list], gen_sp)

    # 组装所有 (step, sample) 的判决 prompt
    vprompts, vmap = [], []
    for i, o in enumerate(gout):
        for si, samp in enumerate(o.outputs):
            vprompts.append({"prompt_token_ids": base_list[i] + list(samp.token_ids[:a.K]) + cue_list[i]})
            vmap.append(i)
    vsp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    vout = llm.generate(vprompts, vsp)

    svals = [[] for _ in meta]
    for (i, o) in zip(vmap, vout):
        lp = o.outputs[0].logprobs[0]
        py = sum(math.exp(v.logprob) for tid, v in lp.items() if tid in yes_ids)
        pn = sum(math.exp(v.logprob) for tid, v in lp.items() if tid in no_ids)
        svals[i].append(math.log(py + 1e-12) - math.log(pn + 1e-12))
    for i, m in enumerate(meta):
        m["svals"] = svals[i]
        m["s_mean"] = sum(svals[i]) / len(svals[i])
    with open(a.out, "a", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    labs = [m["is_err"] for m in meta]
    print(f"\n===== SC think-measure (N={len(meta)}, 错步={sum(labs)}, K={a.K}) =====", flush=True)
    print("  采样数 → AUROC(前 j 条平均):", flush=True)
    for j in range(1, a.samples + 1):
        el = [-(sum(m["svals"][:j]) / j) for m in meta]
        print(f"    samples={j}  AUROC={auroc(el, labs):.3f}", flush=True)
    if a.genprm and os.path.exists(a.genprm):
        gp = {json.loads(l)['id']: json.loads(l) for l in open(a.genprm, encoding='utf-8') if l.strip()}
        ov = [m for m in meta if gp.get(m['cid']) and m['t'] < len(gp[m['cid']]['scores'])]
        if ov:
            ovl = [m['is_err'] for m in ov]
            gel = [1 - gp[m['cid']]['scores'][m['t']] for m in ov]
            print(f"\n===== 同底物重叠 {len(ov)} 步(错{sum(ovl)}) 头对头 =====", flush=True)
            print(f"  GenPRM-1.5B                AUROC={auroc(gel, ovl):.3f}", flush=True)
            for j in range(1, a.samples + 1):
                el = [-(sum(m['svals'][:j]) / j) for m in ov]
                print(f"  ours SC s={j}               AUROC={auroc(el, ovl):.3f}", flush=True)


if __name__ == "__main__":
    main()
