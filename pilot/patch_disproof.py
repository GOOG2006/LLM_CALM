"""创建 disproof 变体:系统提示强制'判incorrect必须指出具体错误,否则judge correct'。
只改系统提示,不改输出格式。基于 prm_evaluate_adaptive5.py 生成 _disproof.py。"""
S='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5_disproof.py'
s=open(S,encoding='utf-8').read()
old="'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'"
new=("'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step. "
     "Judge a paragraph as incorrect ONLY if you can identify a specific, concrete error in it "
     "(a wrong calculation, a false statement, or an invalid deduction) and state exactly what that error is. "
     "If you cannot point to a specific concrete error, judge the paragraph correct.'")
assert s.count(old)==1, ('count', s.count(old))
s=s.replace(old,new)
open(D,'w',encoding='utf-8').write(s)
print('disproof variant written; ok:', 'specific, concrete error' in s)
