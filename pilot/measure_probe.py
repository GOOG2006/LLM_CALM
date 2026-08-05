"""
Verify-by-Measurement 迭代2:自探针正确性 logit(测正确性信念,非流畅度)。
给模型看完整解答,问"第 t 步对吗?",**只读下一 token 的 P(Yes)/P(No) logit,不生成 CoT**。
= GenPRM 判决的"去生成"测量版:一次前向、确定性。
测量量: s_probe = logP(Yes) - logP(No);error-likelihood = -s_probe。
用法: python measure_probe.py --model <path> --n_err 30 --n_cor 30 --genprm out_math10.jsonl
"""
import os, json, argparse
from statistics import mean
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PB_DIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SUBSETS = ["math"]
SYS = "You are a math teacher. Review the solution and judge whether the specified step is correct."


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


def tok_ids(tok, words):
    ids = set()
    for w in words:
        for v in (w, " " + w):
            e = tok(v, add_special_tokens=False).input_ids
            if e:
                ids.add(e[0])
    return list(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_err", type=int, default=30)
    ap.add_argument("--n_cor", type=int, default=30)
    ap.add_argument("--genprm", default=None)
    ap.add_argument("--out", default="results_probe.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    yes_ids = tok_ids(tok, ["Yes", "yes", "YES", "Correct", "correct"])
    no_ids = tok_ids(tok, ["No", "no", "NO", "Incorrect", "incorrect", "Wrong", "wrong"])

    err, cor = load_cases(a.n_err, a.n_cor)
    cases = err + cor
    print(f"cases: err={len(err)} cor={len(cor)}", flush=True)

    rows = []
    for (sub, cid, q, steps, lab) in cases:
        n = len(steps)
        sol = "\n".join(f"Step {j+1}: {steps[j]}" for j in range(n))
        for t in range(n):
            if lab >= 0 and t > lab:
                continue
            is_err = int(lab >= 0 and t == lab)
            user = (f"Question: {q}\n\nSolution:\n{sol}\n\n"
                    f"Is Step {t+1} correct? Reply with only Yes or No.")
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if len(ids) > 4000:
                continue
            inp = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                logits = model(inp).logits[0, -1].float()
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            pyes = float(sum(p[i] for i in yes_ids))
            pno = float(sum(p[i] for i in no_ids))
            s_probe = (torch.log(torch.tensor(pyes + 1e-12)) - torch.log(torch.tensor(pno + 1e-12))).item()
            rows.append(dict(cid=cid, t=t, is_err=is_err, chain_len=n,
                             pyes=pyes, pno=pno, s_probe=s_probe))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [r["is_err"] for r in rows]
    # error-likelihood = -s_probe(Yes 越低越像错);也报 softmax_yes = pyes/(pyes+pno) 的反向
    el = [-r["s_probe"] for r in rows]
    sm = [-(r["pyes"] / (r["pyes"] + r["pno"] + 1e-12)) for r in rows]
    print(f"\n===== 自探针正确性 (N={len(rows)}, 错步={sum(labels)}) =====", flush=True)
    print(f"  s_probe=logP(Yes)-logP(No)   AUROC={auroc(el, labels):.3f}", flush=True)
    print(f"  softmax_yes                  AUROC={auroc(sm, labels):.3f}", flush=True)

    if a.genprm and os.path.exists(a.genprm):
        gp = {json.loads(l)["id"]: json.loads(l) for l in open(a.genprm, encoding="utf-8") if l.strip()}
        gel, gl, oel = [], [], []
        for r in rows:
            g = gp.get(r["cid"])
            if g and r["t"] < len(g["scores"]):
                gel.append(1 - g["scores"][r["t"]]); gl.append(r["is_err"]); oel.append(-r["s_probe"])
        if gl:
            print(f"\n  [重叠 {len(gl)} 步] GenPRM AUROC={auroc(gel, gl):.3f}   "
                  f"probe AUROC={auroc(oel, gl):.3f}", flush=True)


if __name__ == "__main__":
    main()
