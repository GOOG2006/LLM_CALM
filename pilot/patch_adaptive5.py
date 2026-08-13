"""adaptive5:精准路由(仅"完整算式 num op num = num"的步用代码,代码能真核验)+ 报错退避。
基于官方 prm_evaluate.py 生成 prm_evaluate_adaptive5.py。"""
SRC = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate.py"
DST = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py"
s = open(SRC, encoding="utf-8").read()

ROUTER = r'''
import re as _re
_NUM = r'\d[\d,]*\.?\d*'
_COMPUTE = _re.compile(_NUM + r'\s*[\+\-\*/×·÷]\s*' + _NUM + r'.{0,25}?=\s*' + _NUM)
def route_use_code(t):
    """仅当该步含完整可核验算术(数字 运算 数字 ... = 数字)才用代码。"""
    return bool(_COMPUTE.search(t or ''))
'''

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, f"expected {n} of <<{old[:45]}>> got {s.count(old)}"
    s = s.replace(old, new)

rep("os.environ['VLLM_USE_V1'] = '0'\n", "os.environ['VLLM_USE_V1'] = '0'\n" + ROUTER)
rep("        step_scores = []\n", "        step_scores = []\n        used_code_list = []\n        code_enabled = True\n")
rep("            paths = conversation[:step_index]\n            cur_step += 1\n",
    "            paths = conversation[:step_index]\n            cur_step += 1\n"
    "            use_code_this_step = code_enabled and route_use_code(conversation[step_index-1].get('content','') if step_index>0 else '')\n")
rep("                analyze=args.analyze,\n                verify=args.verify,\n                execute=args.execute,\n",
    "                analyze=args.analyze,\n                verify=use_code_this_step,\n                execute=use_code_this_step,\n")
rep("            step_scores.append(reward)\n",
    "            step_scores.append(reward)\n            used_code_list.append(use_code_this_step)\n"
    "            if use_code_this_step and ('Code execute Error' in outputs[0] or 'Code format error' in outputs[0]):\n"
    "                code_enabled = False\n")
rep("        data_new['value'] = step_scores\n", "        data_new['value'] = step_scores\n        data_new['used_code'] = used_code_list\n")

open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST, "| ok:", "route_use_code" in s and "code_enabled = False" in s)
