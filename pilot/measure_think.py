"""
Verify-by-Measurement 迭代4:think-then-measure。
先让模型生成一小段推理(贪心,无代码),再**只读 Yes/No 的 logit**(测量,不解析生成判决)。
扫推理长度 K,看多短的思考能把测量拉到 GenPRM 水平。攻 GenPRM 的 W3/W4/W7(免代码/更短)。
一次生成 256 token,复用其前缀评多个 K。
用法: python measure_think.py --model <path> --n_err 5 --n_cor 5 --genprm out_math10.jsonl
"""
import os, json, argparse
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PB_DIR = "/root/autodl-tmp/genprm_work/ProcessBench"
SUBSETS = ["math"]
SYS = "You are a math teacher. Analyze whether the specified step is correct."
KS = [128, 256, 512, 768]


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
    ap.add_argument("--n_err", type=int, default=5)
    ap.add_argument("--n_cor", type=int, default=5)
    ap.add_argument("--genprm", default=None)
    ap.add_argument("--out", default="results_think.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    yes_ids = tok_ids(tok, ["Yes", "yes", "Correct", "correct"])
    no_ids = tok_ids(tok, ["No", "no", "Incorrect", "incorrect", "Wrong", "wrong"])
    cue = tok("\n\nFinal judgment — is this step correct? Answer with only Yes or No.\nAnswer:",
              add_special_tokens=False).input_ids

    err, cor = load_cases(a.n_err, a.n_cor)
    cases = err + cor
    print(f"cases: err={len(err)} cor={len(cor)}", flush=True)

    rows = []
    fout = open(a.out, "w", encoding="utf-8")

    def running_report():
        labs = [r["is_err"] for r in rows]
        msg = f"[进度 {len(rows)}步 错{sum(labs)}] " + " ".join(
            f"K{K}={auroc([-r[f's{K}'] for r in rows], labs):.3f}" for K in KS)
        print(msg, flush=True)

    for (cid, q, steps, lab) in cases:
        n = len(steps)
        sol = "\n".join(f"Step {j+1}: {steps[j]}" for j in range(n))
        for t in range(n):
            if lab >= 0 and t > lab:
                continue
            is_err = int(lab >= 0 and t == lab)
            user = (f"Question: {q}\n\nSolution:\n{sol}\n\n"
                    f"Analyze whether Step {t+1} is correct.")
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
            base_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if len(base_ids) > 3200:
                continue
            inp = torch.tensor([base_ids], device="cuda")
            with torch.no_grad():
                gen = model.generate(inp, max_new_tokens=max(KS), do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            reason = gen[0, len(base_ids):].tolist()
            rec = dict(cid=cid, t=t, is_err=is_err, chain_len=n)
            for K in KS:
                ctx = base_ids + reason[:K] + cue
                with torch.no_grad():
                    logits = model(torch.tensor([ctx], device="cuda")).logits[0, -1].float()
                p = F.softmax(logits, dim=-1)
                pyes = float(sum(p[i] for i in yes_ids)); pno = float(sum(p[i] for i in no_ids))
                rec[f"s{K}"] = torch.log(torch.tensor(pyes + 1e-12)).item() - torch.log(torch.tensor(pno + 1e-12)).item()
            rows.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            if len(rows) % 12 == 0 and sum(r["is_err"] for r in rows) >= 2:
                running_report()
    fout.close()

    labels = [r["is_err"] for r in rows]
    print(f"\n===== think-then-measure (N={len(rows)}, 错步={sum(labels)}) =====", flush=True)
    for K in KS:
        au = auroc([-r[f"s{K}"] for r in rows], labels)  # Yes 越低越像错
        print(f"  K={K:>3} 推理token后读Yes/No   AUROC={au:.3f}", flush=True)

    if a.genprm and os.path.exists(a.genprm):
        gp = {json.loads(l)["id"]: json.loads(l) for l in open(a.genprm, encoding="utf-8") if l.strip()}
        gel, gl = [], []
        for r in rows:
            g = gp.get(r["cid"])
            if g and r["t"] < len(g["scores"]):
                gel.append(1 - g["scores"][r["t"]]); gl.append(r["is_err"])
        if gl:
            print(f"\n  [重叠{len(gl)}步] GenPRM AUROC={auroc(gel, gl):.3f}", flush=True)


if __name__ == "__main__":
    main()
