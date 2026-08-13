"""smart router 变体:把代码路由从"显式算术"扩到"概念化的可计算断言"
(LCM/GCD、Σ/求和、阶乘/多项式系数、分数方程/交叉相乘、几何等式 PA=PB)。
判别在手挑案上验证:6个可计算断言全命中、2个真抽象不误开。
基于 prm_evaluate_adaptive5.py 生成 _smart.py。"""
S='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5_smart.py'
s=open(S,encoding='utf-8').read()

old = "    return bool(_COMPUTE.search(t or ''))"
new = (
"    t = t or ''\n"
"    if _COMPUTE.search(t):\n"
"        return True\n"
"    # 可计算断言(概念化但机械可验)\n"
"    if _re.search(r'\\b(lcm|gcd)\\b|\\\\sum|\\bsum_|summation|\\bbinomial|\\bmultinomial|\\bfactorial|\\bchoose\\b|inclusion|\\bmodulo\\b', t, _re.I):\n"
"        return True\n"
"    if _re.search(r'\\\\frac\\{[^}]*\\}\\{[^}]*\\}\\s*=|=\\s*\\\\frac|cross-multiply', t):\n"
"        return True\n"
"    if _re.search(r'\\d\\s*!|n!|k!', t):\n"
"        return True\n"
"    if _re.search(r'[A-Z]{2}\\s*=\\s*[A-Z]{2}', t):\n"
"        return True\n"
"    return False"
)
assert s.count(old)==1, ('count', s.count(old))
s=s.replace(old,new)
open(D,'w',encoding='utf-8').write(s)
import ast
ast.parse(s)
print('smart variant written & syntax OK; has smart patterns:', 'lcm|gcd' in s and 'cross-multiply' in s)
