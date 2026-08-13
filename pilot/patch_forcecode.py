"""强制截断 analyze 进入代码阶段:路由到代码(verify=True)的步,
把 analyze 干净截到 </analyze> 再拼 verify_start,确保代码阶段一定运行(不被提前的判决跳过)。
从干净备份 .bak_forcecode 生成。"""
BAK='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py.bak_forcecode'
DST='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py'
s=open(BAK,encoding='utf-8').read()
old="                cur_prompt = analyze_start + output1.text + verify_start  # generate <verify> if verify is True"
# 构造替换:analyze_start + output1.text.split('</analyze>')[0] + '</analyze>\n' + verify_start
new=("                cur_prompt = analyze_start + output1.text.split('</analyze>')[0] + "
     + repr('</analyze>\n')
     + " + verify_start  # force-truncate analyze, ensure code stage runs")
assert s.count(old)==1, ('count', s.count(old))
s=s.replace(old,new)
open(DST,'w',encoding='utf-8').write(s)
import ast
ast.parse(s)
print('OK force-code patched; new line:')
for ln in s.splitlines():
    if 'force-truncate analyze' in ln: print('   ', ln.strip())
