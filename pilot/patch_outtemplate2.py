"""从 .bak_outtmpl(原版)恢复,应用'参考并仔细分析'的判决模板(而非'严格按照')。
让判决批判性权衡 analyze/code,而不是盲从过度挑刺的 analyze。"""
pairs = [
 ('/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py.bak_outtmpl',
  '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'),
 ('/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py.bak_outtmpl',
  '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py'),
]
OLD = r'<output>\n**Judgement**: $\\boxed'
NEW = r'<output>\n**Judgement** (after carefully analyzing and critically weighing whether the analysis and code output above truly reveal an error): $\\boxed'
total=0
for bak, dst in pairs:
    s=open(bak,encoding='utf-8').read()  # 从原版备份读
    c=s.count(OLD)
    s=s.replace(OLD,NEW)
    open(dst,'w',encoding='utf-8').write(s)
    total+=c
    print(dst,'replaced',c)
import ast
for _,dst in pairs: ast.parse(open(dst,encoding='utf-8').read())
print('total',total,'| syntax OK')
