"""改 output 模板:让 boxed 判决显式基于 analyze 和 code 输出。
把 '<output>\\n**Judgement**: $\\boxed' 改成
'<output>\\n**Judgement** (based strictly on the analysis and code output above): $\\boxed'
改 eval 的 arg 默认 + genprm_inference 的默认(3处)。"""
import re
files = [
 '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py',
 '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py',
]
OLD = r'<output>\n**Judgement**: $\\boxed'
NEW = r'<output>\n**Judgement** (based strictly on the analysis and code output above): $\\boxed'
total=0
for f in files:
    s=open(f,encoding='utf-8').read()
    c=s.count(OLD)
    s=s.replace(OLD,NEW)
    open(f,'w',encoding='utf-8').write(s)
    total+=c
    print(f, 'replaced', c)
# 语法检查
import ast
for f in files: ast.parse(open(f,encoding='utf-8').read())
print('total replaced', total, '| syntax OK')
