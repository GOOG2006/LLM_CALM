"""adaptive3:默认全程用代码 + 一击退避(去掉启发式门槛)。
代码在能算的子集全开(拿满配好处),报错就对该题剩余退避(拿去代码好处)。
基于官方 prm_evaluate.py 生成 prm_evaluate_adaptive3.py。"""
SRC = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate.py"
DST = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive3.py"
s = open(SRC, encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, f"expected {n} of <<{old[:45]}>> got {s.count(old)}"
    s = s.replace(old, new)

# 状态初始化
rep("        step_scores = []\n",
    "        step_scores = []\n        used_code_list = []\n        code_enabled = True\n")
# 每步:直接用 code_enabled(无启发式)
rep("            paths = conversation[:step_index]\n            cur_step += 1\n",
    "            paths = conversation[:step_index]\n            cur_step += 1\n"
    "            use_code_this_step = code_enabled\n")
# inference 逐步 verify/execute
rep("                analyze=args.analyze,\n                verify=args.verify,\n                execute=args.execute,\n",
    "                analyze=args.analyze,\n                verify=use_code_this_step,\n                execute=use_code_this_step,\n")
# 反馈退避
rep("            step_scores.append(reward)\n",
    "            step_scores.append(reward)\n"
    "            used_code_list.append(use_code_this_step)\n"
    "            if use_code_this_step and ('Code execute Error' in outputs[0] or 'Code format error' in outputs[0]):\n"
    "                code_enabled = False\n")
# 保存
rep("        data_new['value'] = step_scores\n",
    "        data_new['value'] = step_scores\n        data_new['used_code'] = used_code_list\n")

open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST, "| default-code+backoff:", "use_code_this_step = code_enabled" in s and "code_enabled = False" in s)
