"""把官方 prm_evaluate.py 改造成 per-step 自适应代码路由版 prm_evaluate_adaptive.py。
每步用启发式路由决定是否用代码(计算型→带代码; 概念型→analyze-only)。远程运行。"""
import io

SRC = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate.py"
DST = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive.py"
s = open(SRC, encoding="utf-8").read()

ROUTER = '''
import re as _re
_NUM = r'-?\\d+(?:\\.\\d+)?'
_ARITH = _re.compile(_NUM + r'\\s*[\\+\\-\\*/\\u00d7\\u00f7]\\s*' + _NUM)
_EQNUM = _re.compile(r'=\\s*' + _NUM)
def route_use_code(t):
    """启发式:含'数字 运算符 数字'或'= 数字'的计算型步骤才用代码。"""
    t = t or ''
    return bool(_ARITH.search(t) or _EQNUM.search(t))
'''

def rep(old, new, s, n=1):
    c = s.count(old)
    assert c == n, f"expected {n} of <<{old[:50]}...>> got {c}"
    return s.replace(old, new)

# R1: 注入 router(在 VLLM_USE_V1 之后)
s = rep("os.environ['VLLM_USE_V1'] = '0'\n", "os.environ['VLLM_USE_V1'] = '0'\n" + ROUTER, s)

# R2: 每步计算 use_code(在 cur_step += 1 之后)
s = rep("            paths = conversation[:step_index]\n            cur_step += 1\n",
        "            paths = conversation[:step_index]\n            cur_step += 1\n"
        "            use_code_this_step = route_use_code(conversation[step_index-1].get('content','') if step_index>0 else '')\n", s)

# R3: inference 的 verify/execute 改为逐步
s = rep("                analyze=args.analyze,\n                verify=args.verify,\n                execute=args.execute,\n",
        "                analyze=args.analyze,\n                verify=use_code_this_step,\n                execute=use_code_this_step,\n", s)

# R4: 初始化 used_code_list
s = rep("        step_scores = []\n", "        step_scores = []\n        used_code_list = []\n", s)

# R5: 记录 used_code
s = rep("            step_scores.append(reward)\n",
        "            step_scores.append(reward)\n            used_code_list.append(use_code_this_step)\n", s)

# R6: 保存 used_code
s = rep("        data_new['value'] = step_scores\n",
        "        data_new['value'] = step_scores\n        data_new['used_code'] = used_code_list\n", s)

open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST)
print("router present:", "route_use_code" in s, "| per-step verify:", "verify=use_code_this_step" in s)
