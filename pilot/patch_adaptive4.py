"""adaptive4:per-step 模型路由。每步先让模型判"验证这步用代码有用吗(数值计算→Yes/概念证明→No)",
读 Yes/No logit 决定是否用代码。跨数据集统一、内容驱动的真自适应。基于官方 prm_evaluate.py。"""
SRC = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate.py"
DST = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive4.py"
s = open(SRC, encoding="utf-8").read()

ROUTER = '''
import math as _math
from vllm import SamplingParams as _RSP
_RSAMP = _RSP(temperature=0.0, max_tokens=1, logprobs=20)
_YESNO = {}
_ROUTE_PROMPT = ("You are checking ONE step of a math solution for errors. "
    "Is executing Python code useful to verify THIS step? Answer 'Yes' only if it is a concrete "
    "numeric or algebraic computation that code can check; answer 'No' if it is a conceptual, "
    "logical, definitional, or proof step.\\n\\nStep: {step}\\n\\nAnswer (Yes or No):")
def route_use_code(genprm, step_text):
    if not _YESNO:
        for w in ['Yes', 'yes', 'No', 'no']:
            e = genprm.tokenizer(w, add_special_tokens=False).input_ids
            if e: _YESNO[w] = e[0]
    prompt = _ROUTE_PROMPT.format(step=(step_text or '')[:1500])
    try:
        out = genprm.model.generate([prompt], _RSAMP, use_tqdm=False)[0].outputs[0]
        lp = out.logprobs[0] if out.logprobs else None
    except Exception:
        return True
    if not lp: return True
    py = sum(_math.exp(v.logprob) for tid, v in lp.items() if tid in (_YESNO.get('Yes'), _YESNO.get('yes')))
    pn = sum(_math.exp(v.logprob) for tid, v in lp.items() if tid in (_YESNO.get('No'), _YESNO.get('no')))
    return py >= pn
'''

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, f"expected {n} of <<{old[:45]}>> got {s.count(old)}"
    s = s.replace(old, new)

rep("os.environ['VLLM_USE_V1'] = '0'\n", "os.environ['VLLM_USE_V1'] = '0'\n" + ROUTER)
rep("        step_scores = []\n", "        step_scores = []\n        used_code_list = []\n")
rep("            paths = conversation[:step_index]\n            cur_step += 1\n",
    "            paths = conversation[:step_index]\n            cur_step += 1\n"
    "            use_code_this_step = route_use_code(genprm, conversation[step_index-1].get('content','') if step_index>0 else '')\n")
rep("                analyze=args.analyze,\n                verify=args.verify,\n                execute=args.execute,\n",
    "                analyze=args.analyze,\n                verify=use_code_this_step,\n                execute=use_code_this_step,\n")
rep("            step_scores.append(reward)\n",
    "            step_scores.append(reward)\n            used_code_list.append(use_code_this_step)\n")
rep("        data_new['value'] = step_scores\n",
    "        data_new['value'] = step_scores\n        data_new['used_code'] = used_code_list\n")

open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST, "| model-router:", "route_use_code(genprm" in s)
