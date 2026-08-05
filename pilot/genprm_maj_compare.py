"""
SOTA确认(K=2): GenPRM-1.5B 上 朴素Maj@2 vs 去相关Maj@2,等算力(每步各2次)。
ArmA: 默认模板 ×2 (temperature 采样)
ArmB: 2 个方法多样视角 各1次 (题面独立重算 / 实体核对)
记每步 4 个单次 reward,算 F1 + 视角误差相关度。
用法: python genprm_maj_compare.py --n 30
"""
import os, sys, json, argparse
sys.path.append('/root/autodl-tmp/genprm_work/GenPRM/src')
os.environ['VLLM_USE_V1'] = '0'
from prm_evaluation.genprm_inference import GenPRM, CodeExecutor

SYS = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."
MODEL = "/root/autodl-tmp/genprm_work/models/GenPRM-1.5B"
PREDS = "/root/autodl-tmp/genprm_work/genprm_preds.json"

DEF_A = "<analyze>\nLet's analyze the Paragraph {cur_step} step by step: "
DEF_V = "<verify>\nLet's use python code to find any potential error:\n```python\n"
# 去相关视角1: 题面独立重算(打操作数/变量绑定错)
V1_A = "<analyze>\nLet me recompute Paragraph {cur_step} using ONLY the numbers stated in the original problem, not the variables defined by the solution: "
V1_V = "<verify>\nLet's recompute this step's value directly from the original problem's given numbers with code (do not reuse the solution's substitutions):\n```python\n"
# 去相关视角2: 实体/读题核对(打题意误读)
V2_A = "<analyze>\nLet me re-read the problem carefully and check whether Paragraph {cur_step} uses the correct quantity for the correct entity/time and interprets the wording correctly: "
V2_V = "<verify>\nLet's use code to check whether the quantities used in this step match the correct entities described in the problem:\n```python\n"

def build_paths(problem, steps, gold_step_1idx):
    di = list(steps); di[0] = problem + "\n" + di[0]
    conv = [{"role": "system", "content": SYS}]
    for s in di:
        conv.append({"role": "user", "content": s}); conv.append({"role": "assistant", "content": ""})
    asst = [k for k, m in enumerate(conv) if m["role"] == "assistant"]
    return conv, asst

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); a = ap.parse_args()
    data = json.load(open(PREDS, encoding="utf-8"))
    err = [d for d in data if d["label"] != -1][: a.n // 2]
    cor = [d for d in data if d["label"] == -1][: a.n - a.n // 2]
    sub = err + cor
    print(f"{len(sub)} samples ({len(err)} err/{len(cor)} cor)", flush=True)
    genprm = GenPRM(MODEL, tensor_parallel_size=1); ce = CodeExecutor()

    def rew(paths, cur, aT, vT):
        _, r = genprm.inference(paths, majority_num=1, cur_step=cur, code_executor=ce,
                                analyze_template=aT, verify_template=vT, logging=False)
        return float(r)

    rows = []
    for si, d in enumerate(sub):
        steps = d["steps"]; conv, asst = build_paths(d["problem"], steps, None)
        per = []
        for i in range(len(steps)):
            paths = conv[: asst[i]]; cur = i + 1
            a1 = rew(paths, cur, DEF_A, DEF_V); a2 = rew(paths, cur, DEF_A, DEF_V)      # 朴素×2
            b1 = rew(paths, cur, V1_A, V1_V); b2 = rew(paths, cur, V2_A, V2_V)          # 去相关2视角
            per.append([a1, a2, b1, b2])
        rows.append({"label": d["label"], "genprm_pred": d["genprm_pred"], "per": per})
        print(f"[{si+1}/{len(sub)}] label={d['label']} steps={len(steps)}", flush=True)
        json.dump(rows, open("/root/autodl-tmp/genprm_work/maj_compare_out.json", "w"))
    report(rows)

def pred_from(per, cols):
    for i, p in enumerate(per):
        avg = sum(p[c] for c in cols) / len(cols)
        if avg < 0.5:
            return i
    return -1

def report(rows):
    err = [r for r in rows if r["label"] != -1]; cor = [r for r in rows if r["label"] == -1]
    def f1of(cols, name):
        ae = sum(pred_from(r["per"], cols) == r["label"] for r in err) / max(len(err), 1)
        ac = sum(pred_from(r["per"], cols) == -1 for r in cor) / max(len(cor), 1)
        f1 = 2 * ae * ac / max(ae + ac, 1e-9)
        print(f"{name:<24} eacc={ae:.3f} cacc={ac:.3f} F1={f1*100:.1f}", flush=True)
    print("\n===== K=2 SOTA确认 (GenPRM-1.5B, 等算力) =====", flush=True)
    f1of([0, 1], "朴素 Maj@2 (default×2)")
    f1of([2, 3], "去相关 Maj@2 (2视角)")
    # 视角误差相关度: 两次 reward 差的绝对值均值(越大=越去相关)。粗略机制指标
    import statistics
    da = statistics.mean(abs(p[0] - p[1]) for r in rows for p in r["per"])
    db = statistics.mean(abs(p[2] - p[3]) for r in rows for p in r["per"])
    print(f"\n视角间 reward 分歧(越大越去相关): 朴素={da:.3f}  去相关={db:.3f}", flush=True)

if __name__ == "__main__":
    main()
