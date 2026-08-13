"""adaptive6:默认全用代码 + 代码输出质量退避。
代码不只"报错"退避,连"跑出来不是干净数字/空"也退避(olympiad 证明步代码常输出非数值)。
→ omnimath(代码输出数字)保持用代码;olympiad(输出无关/非数值)退避到无代码。同一机制。
基于官方 prm_evaluate.py 生成 prm_evaluate_adaptive6.py。"""
SRC = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate.py"
DST = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive6.py"
s = open(SRC, encoding="utf-8").read()

HELP = r'''
import re as _re6
def _bad_code_output(text):
    """判断该步代码是否'没用':无代码输出 / 空 / 报错 / 无数字 → 视为无效。"""
    blocks = _re6.findall(r'\[Code Output\]\s*```(.*?)```', text or '', _re6.DOTALL)
    if not blocks:
        return True
    out = blocks[-1].strip()
    if not out:
        return True
    if 'Error' in out or 'error' in out or 'Traceback' in out:
        return True
    if not _re6.search(r'\d', out):
        return True
    return False
'''

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, f"expected {n} of <<{old[:45]}>> got {s.count(old)}"
    s = s.replace(old, new)

rep("os.environ['VLLM_USE_V1'] = '0'\n", "os.environ['VLLM_USE_V1'] = '0'\n" + HELP)
rep("        step_scores = []\n", "        step_scores = []\n        used_code_list = []\n        code_enabled = True\n")
rep("            paths = conversation[:step_index]\n            cur_step += 1\n",
    "            paths = conversation[:step_index]\n            cur_step += 1\n"
    "            use_code_this_step = code_enabled\n")
rep("                analyze=args.analyze,\n                verify=args.verify,\n                execute=args.execute,\n",
    "                analyze=args.analyze,\n                verify=use_code_this_step,\n                execute=use_code_this_step,\n")
rep("            step_scores.append(reward)\n",
    "            step_scores.append(reward)\n            used_code_list.append(use_code_this_step)\n"
    "            if use_code_this_step and _bad_code_output(outputs[0]):\n"
    "                code_enabled = False  # 代码无效(报错/非数值/空)→ 该题剩余退避到无代码\n")
rep("        data_new['value'] = step_scores\n", "        data_new['value'] = step_scores\n        data_new['used_code'] = used_code_list\n")

open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST, "| ok:", "_bad_code_output" in s and "use_code_this_step = code_enabled" in s)
