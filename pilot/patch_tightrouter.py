"""tight router:只保留最机械、olympiad上不常见的可计算断言模式。
v1算式 + lcm/gcd + 分数方程/交叉相乘 + 阶乘/多项式系数。
砍掉 sum/Σ(证明常见)、mod(数论证明常见)、inclusion、几何等式(太宽/需概念)。
基于 prm_evaluate_adaptive5.py 生成 _tight.py。"""
S='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5_tight.py'
s=open(S,encoding='utf-8').read()
old = "    return bool(_COMPUTE.search(t or ''))"
new = (
"    t = t or ''\n"
"    if _COMPUTE.search(t):\n"
"        return True\n"
"    if _re.search(r'\\b(lcm|gcd)\\b|\\bbinomial|\\bmultinomial|\\bchoose\\b', t, _re.I):\n"
"        return True\n"
"    if _re.search(r'\\\\frac\\{[^}]*\\}\\{[^}]*\\}\\s*=|=\\s*\\\\frac|cross-multiply', t):\n"
"        return True\n"
"    if _re.search(r'\\d\\s*!|n!|k!', t):\n"
"        return True\n"
"    return False"
)
assert s.count(old)==1, ('count', s.count(old))
s=s.replace(old,new)
open(D,'w',encoding='utf-8').write(s)
import ast; ast.parse(s)
print('tight variant written & syntax OK')
