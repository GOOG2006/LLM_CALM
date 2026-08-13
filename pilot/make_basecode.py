"""生成 prm_evaluate_basecode.py = adaptive5 但强制每步开码(route_use_code 恒 True)。"""
S = '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D = '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_basecode.py'
s = open(S, encoding='utf-8').read()
old = "    return bool(_COMPUTE.search(t or ''))"
new = "    return True  # basecode: force code on every step"
assert s.count(old) == 1, ('count', s.count(old))
s = s.replace(old, new)
open(D, 'w', encoding='utf-8').write(s)
import ast; ast.parse(s)
print('wrote basecode variant, syntax OK')
