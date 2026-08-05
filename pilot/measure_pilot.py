"""
Verify-by-Measurement — 富测量算子 pilot(HF 前向,一次拿全分布)。
对 ProcessBench 每步算多个训练自由测量量,报每个的 error-AUROC,并与 GenPRM 头对头。
测量量(均来自 [prefix+step] 一次前向,对 step 的 token 位置聚合):
  mean_logp / min_logp        : 前向 logprob(负 perplexity)
  mean_ent  / max_ent         : 每步 token 熵(全词表)
  mean_margin / max_margin    : argmax_logprob - actual_logprob(模型与所写 token 的分歧)
用法: python measure_pilot.py --model <path> --n_err 5 --n_cor 5 --genprm out_math10.jsonl
"""
import os, json, argparse
from statistics import mean, pstdev
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PB_DIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SUBSETS = ["math"]
MAXLEN = 3500


def load_cases(n_err, n_cor):
    err, cor = [], []
    for sub in SUBSETS:
        p = os.path.join(PB_DIR, f"{sub}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            lab = int(r["label"]); steps = r["steps"]
            if not isinstance(steps, list) or len(steps) < 2:
                continue
            (cor if lab == -1 else err).append((sub, r["id"], r["problem"], steps, lab))
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


def best_auroc(scores, labels):
    a = auroc(scores, labels); b = auroc([-x for x in scores], labels)
    return (max(a, b), "raw(大=错)" if a >= b else "反号(小=错)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_err", type=int, default=5)
    ap.add_argument("--n_cor", type=int, default=5)
    ap.add_argument("--genprm", default=None)
    ap.add_argument("--out", default="results_measure_math10.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()

    err, cor = load_cases(a.n_err, a.n_cor)
    cases = err + cor
    print(f"cases: err={len(err)} cor={len(cor)}", flush=True)

    KEYS = ["mean_logp", "min_logp", "mean_ent", "max_ent", "ent_p90", "ent_var", "mean_margin", "max_margin"]
    rows = []
    for (sub, cid, q, steps, lab) in cases:
        n = len(steps)
        for t in range(n):
            if lab >= 0 and t > lab:
                continue  # 污染区
            is_err = int(lab >= 0 and t == lab)
            prev = "".join(f"Step {j+1}: {steps[j]}\n" for j in range(t)) or "(no previous steps)\n"
            prefix = f"Problem:\n{q}\n\nStep-by-step solution:\n{prev}Step {t+1}: "
            step_txt = steps[t]
            pre_ids = tok(prefix, add_special_tokens=True).input_ids
            stp_ids = tok(step_txt, add_special_tokens=False).input_ids
            if len(stp_ids) == 0:
                continue
            ids = pre_ids + stp_ids
            if len(ids) > MAXLEN:
                keep = max(1, MAXLEN - len(stp_ids))
                ids = pre_ids[-keep:] + stp_ids
                pre_len = keep
            else:
                pre_len = len(pre_ids)
            inp = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                logits = model(inp).logits[0]  # [T,V]
            lps, ents, margins = [], [], []
            for i in range(pre_len, len(ids)):
                lg = logits[i - 1].float()
                logp = F.log_softmax(lg, dim=-1)
                p = logp.exp()
                tid = ids[i]
                lps.append(logp[tid].item())
                ents.append(-(p * logp).sum().item())
                margins.append((logp.max() - logp[tid]).item())
            ents_sorted = sorted(ents)
            ent_p90 = ents_sorted[min(len(ents) - 1, int(0.9 * (len(ents) - 1)))]
            ent_var = pstdev(ents) ** 2 if len(ents) > 1 else 0.0
            m = dict(cid=cid, t=t, is_err=is_err, chain_len=n,
                     mean_logp=mean(lps), min_logp=min(lps),
                     mean_ent=mean(ents), max_ent=max(ents), ent_p90=ent_p90, ent_var=ent_var,
                     mean_margin=mean(margins), max_margin=max(margins))
            rows.append(m)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [r["is_err"] for r in rows]
    print(f"\n===== Verify-by-Measurement 算子 error-AUROC (N_steps={len(rows)}, 错步={sum(labels)}) =====", flush=True)
    ranked = []
    for k in KEYS:
        au, d = best_auroc([r[k] for r in rows], labels)
        ranked.append((au, k, d))
    for au, k, d in sorted(ranked, reverse=True):
        print(f"  {k:<12} AUROC={au:.3f}  [{d}]", flush=True)

    # 头对头 GenPRM
    if a.genprm and os.path.exists(a.genprm):
        gp = {}
        for line in open(a.genprm, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); gp[r["id"]] = r
        gp_el, gl = [], []
        for r in rows:
            g = gp.get(r["cid"])
            if g and r["t"] < len(g["scores"]):
                gp_el.append(1.0 - g["scores"][r["t"]]); gl.append(r["is_err"])
        a_gp = auroc(gp_el, gl)
        best = max(ranked)
        print(f"\n  GenPRM (1-score)  AUROC={a_gp:.3f}   |   最佳测量算子 {best[1]}={best[0]:.3f}   Δ={best[0]-a_gp:+.3f}", flush=True)


if __name__ == "__main__":
    main()
