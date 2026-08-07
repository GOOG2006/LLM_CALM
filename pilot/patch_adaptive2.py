"""自适应控制式代码路由 v2:每题内"一击退避"——某步代码报错后,对该题剩余步骤停用代码。
基于官方 prm_evaluate.py 生成 prm_evaluate_adaptive2.py。远程运行。"""
SRC = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate.py"
DST = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive2.py"
s = open(SRC, encoding="utf-8").read()

ROUTER = '''
import re as _re
_NUM = r'-?\\d+(?:\\.\\d+)?'
_ARITH = _re.compile(_NUM + r'\\s*[\\+\\-\\*/\\u00d7\\u00f7]\\s*' + _NUM)
_EQNUM = _re.compile(r'=\\s*' + _NUM)
def route_use_code(t):
    t = t or ''
    return bool(_ARITH.search(t) or _EQNUM.search(t))
'''

def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, f"expected {n} of <<{old[:45]}...>> got {c}"
    s = s.replace(old, new)

# R1 router
rep("os.environ['VLLM_USE_V1'] = '0'\n", "os.environ['VLLM_USE_V1'] = '0'\n" + ROUTER)
# R2 per-problem 状态初始化
rep("        step_scores = []\n",
    "        step_scores = []\n        used_code_list = []\n        code_enabled = True\n")
# R3 每步:use_code = 启用中 且 计算型
rep("            paths = conversation[:step_index]\n            cur_step += 1\n",
    "            paths = conversation[:step_index]\n            cur_step += 1\n"
    "            use_code_this_step = code_enabled and route_use_code(conversation[step_index-1].get('content','') if step_index>0 else '')\n")
# R4 inference 逐步 verify/execute
rep("                analyze=args.analyze,\n                verify=args.verify,\n                execute=args.execute,\n",
    "                analyze=args.analyze,\n                verify=use_code_this_step,\n                execute=use_code_this_step,\n")
# R5 反馈退避:代码报错则对该题剩余停用
rep("            step_scores.append(reward)\n",
    "            step_scores.append(reward)\n"
    "            used_code_list.append(use_code_this_step)\n"
    "            if use_code_this_step and ('Code execute Error' in outputs[0] or 'Code format error' in outputs[0]):\n"
    "                code_enabled = False  # 一击退避:代码不可靠,该题剩余步骤停用代码\n")
# R6 保存
rep("        data_new['value'] = step_scores\n",
    "        data_new['value'] = step_scores\n        data_new['used_code'] = used_code_list\n")

open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST, "| router:", "route_use_code" in s, "| backoff:", "code_enabled = False" in s)
