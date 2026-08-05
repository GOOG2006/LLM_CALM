"""
Detailed-Balance Verifier — Gate 1 存在性 pilot(见 PREREG_detbal.md)。
在 ProcessBench(math+olympiadbench,禁 GSM8K)上,用 DSR1-Distill-Qwen(基座=GenPRM 基座)
对每步算前向可行性 logp_f 与反向可行性 logp_b,σ = logp_f - logp_b(熵产)。

存在性问题: AUROC(σ 作为"该步为错误步"的预测子)。
方向归因闸①: 同时报 AUROC[仅 logp_f] / AUROC[仅 logp_b] / AUROC[σ];
  若 σ 相对"仅 logp_f"(=普通 perplexity)增益 <0.03 → 反向项无独立价值,判死(降级为 perplexity 验证器)。

用法: python detbal_pilot.py --model <path> --n 160 --out results_detbal.jsonl
vLLM prompt_logprobs teacher-forcing,精确 token 边界(prompt_token_ids)。
"""
import os, re, json, argparse, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PB_DIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SUBSETS = ["math"]  # 头对头对比:仅 math(禁 gsm8k dead-4)

# 反向重构提示:给出问题 + 后续步(含答案),要求恢复"更早的一步"。
BWD_HDR = "Problem:\n{q}\n\nThe LATER part of a step-by-step solution is:\n{future}\n\nAn EARLIER step of the same solution was:\n"
FWD_HDR = "Problem:\n{q}\n\nStep-by-step solution:\n{prev}"


def load_cases(n_err, n_cor):
    """取 label>=0(有错)与 label==-1(全对)两类。返回 [(subset,id,problem,steps,label)]."""
    err, cor = [], []
    for sub in SUBSETS:
        p = os.path.join(PB_DIR, f"{sub}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            lab = int(r["label"])
            steps = r["steps"]
            if not isinstance(steps, list) or len(steps) < 2:
                continue
            rec = (sub, r["id"], r["problem"], steps, lab)
            (cor if lab == -1 else err).append(rec)
    return err[:n_err], cor[:n_cor]


def build_pairs(cases):
    """
    为每个 (case, step_t) 生成前向/反向打分任务。
    正例: t == label(首错步)。负例: t < label(已知正确步)或 label==-1 的所有步。
    label 之后的步(污染区)不纳入 AUROC。
    返回: tasks = [dict(cid, sub, t, is_err, chain_len, fwd_prefix, bwd_prefix, target)]
    """
    tasks = []
    for (sub, cid, q, steps, lab) in cases:
        n = len(steps)
        for t in range(n):
            if lab >= 0 and t > lab:
                continue  # 污染区,跳过
            is_err = int(lab >= 0 and t == lab)
            prev = "".join(f"Step {j+1}: {steps[j]}\n" for j in range(t)) or "(no previous steps)\n"
            future = "".join(f"Step {j+1}: {steps[j]}\n" for j in range(t + 1, n)) or "(this was the final step)\n"
            fwd_prefix = FWD_HDR.format(q=q, prev=prev) + f"Step {t+1}: "
            bwd_prefix = BWD_HDR.format(q=q, future=future) + f"Step {t+1}: "
            target = steps[t]
            tasks.append(dict(cid=cid, sub=sub, t=t, is_err=is_err, chain_len=n,
                              fwd_prefix=fwd_prefix, bwd_prefix=bwd_prefix, target=target))
    return tasks


def score_batch(llm, tok, prefixes, targets):
    """teacher-forced 平均 token logprob of target,精确边界(prompt_token_ids)。
    取每个 target 位置**实际 token** 的 logprob(vLLM prompt_logprobs 含真实 token)。"""
    prompts, all_ids, spans = [], [], []
    for pre, tgt in zip(prefixes, targets):
        pre_ids = tok(pre, add_special_tokens=True).input_ids
        tgt_ids = tok(tgt, add_special_tokens=False).input_ids
        if len(tgt_ids) == 0:
            tgt_ids = tok(" ", add_special_tokens=False).input_ids
        ids = pre_ids + tgt_ids
        if len(ids) > 4000:  # 截断保护(< max_model_len),保住 target
            keep = max(1, 4000 - len(tgt_ids))
            ids = pre_ids[-keep:] + tgt_ids
            pre_len = keep
        else:
            pre_len = len(pre_ids)
        prompts.append({"prompt_token_ids": ids})
        all_ids.append(ids)
        spans.append((pre_len, len(ids)))
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    # 分块跑:prompt_logprobs 的全词表 ranks 激活内存 ∝ 批内总 token,必须限批避免 OOM
    CHUNK = 2
    outs = []
    for i in range(0, len(prompts), CHUNK):
        outs.extend(llm.generate(prompts[i:i + CHUNK], sp, use_tqdm=False))
    scores = []
    for o, ids, (s, e) in zip(outs, all_ids, spans):
        pl = o.prompt_logprobs  # list, pl[i] = {tid: Logprob} (含真实 token) 或 None(首 token)
        lps = []
        for i in range(s, e):
            d = pl[i] if (pl is not None and i < len(pl)) else None
            if not d:
                continue
            tid = ids[i]
            if tid in d:
                lps.append(d[tid].logprob)
            else:  # 极少数不在返回集里,用该位置最小 logprob 作保守下界
                lps.append(min(v.logprob for v in d.values()))
        scores.append(sum(lps) / len(lps) if lps else -20.0)
    return scores


def auroc(scores, labels):
    """AUROC:score 越大越像正类(is_err=1)。返回 auroc 与方向(是否需反号)。"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan"), 0
    wins = ties = 0
    for p in pos:
        for ng in neg:
            if p > ng:
                wins += 1
            elif p == ng:
                ties += 1
    a = (wins + 0.5 * ties) / (len(pos) * len(neg))
    return a, len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_err", type=int, default=140)
    ap.add_argument("--n_cor", type=int, default=40)
    ap.add_argument("--out", default="results_detbal.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    err, cor = load_cases(a.n_err, a.n_cor)
    cases = err + cor
    print(f"cases: err={len(err)} cor={len(cor)}", flush=True)
    tasks = build_pairs(cases)
    print(f"step-tasks={len(tasks)} (pos={sum(t['is_err'] for t in tasks)})", flush=True)

    # 打分不生成,KV cache 需求极小 → gmu 压到 0.3,把显存留给 prompt_logprobs 的全词表 ranks 激活
    llm = LLM(model=a.model, dtype="half", max_model_len=4096,
              gpu_memory_utilization=0.30, enforce_eager=True, max_num_seqs=2)

    fwd = score_batch(llm, tok, [t["fwd_prefix"] for t in tasks], [t["target"] for t in tasks])
    bwd = score_batch(llm, tok, [t["bwd_prefix"] for t in tasks], [t["target"] for t in tasks])

    rows = []
    for t, lf, lb in zip(tasks, fwd, bwd):
        r = dict(t); r.pop("fwd_prefix"); r.pop("bwd_prefix"); r.pop("target")
        r["logp_f"] = lf; r["logp_b"] = lb; r["sigma"] = lf - lb
        rows.append(r)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [r["is_err"] for r in rows]
    # 错误步预期:前向可行性更低(logp_f 小) → 用 -logp_f 作“像错误”的分;σ 越大越不可逆→越像错误
    def rep(name, sc):
        a1, npos = auroc(sc, labels)
        a2, _ = auroc([-x for x in sc], labels)
        best = max(a1, a2)
        direction = "raw(大=错)" if a1 >= a2 else "反号(小=错)"
        print(f"{name:<22} AUROC={best:.3f}  [{direction}]  (raw={a1:.3f}, npos={npos})", flush=True)
        return best
    print(f"\n===== Detailed-Balance Gate 1 (N_steps={len(rows)}, 错步={sum(labels)}) =====", flush=True)
    A_f = rep("仅前向 logp_f", [r["logp_f"] for r in rows])
    A_b = rep("仅反向 logp_b", [r["logp_b"] for r in rows])
    A_s = rep("σ = logp_f-logp_b", [r["sigma"] for r in rows])
    print(f"\n闸①方向归因: σ({A_s:.3f}) − 仅前向({A_f:.3f}) = {A_s - A_f:+.3f}  "
          f"(<0.03 → 反向项无独立价值,判死为 perplexity 验证器)", flush=True)
    verdict = "过闸①,进闸②③④" if (A_s >= 0.65 and A_s - A_f >= 0.03) else \
              ("判死(σ<0.55)" if A_s < 0.55 else "灰区/未过闸①")
    print(f"存在性: σ AUROC={A_s:.3f} → {verdict}", flush=True)


if __name__ == "__main__":
    main()
