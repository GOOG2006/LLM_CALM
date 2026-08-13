"""比较式 Stage-2 变体:Stage-1 照常;若出现级联(>=2 个被判No的步),
把候选步交给模型做'比较选择'——选出第一个真错(\\boxed{N}),据此覆盖预测。
只在级联案触发;非级联案不变。基于 prm_evaluate_adaptive5.py 生成 _choose.py。"""
S='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5_choose.py'
s=open(S,encoding='utf-8').read()

anchor = "        data_new['value'] = step_scores\n"
block = '''        # ===== Stage 2: 比较式复核(仅级联 >=2 flag)=====
        _flags = [ _i for _i, _s in enumerate(step_scores) if float(_s) < 0.5 ]
        data_new['value_stage1'] = list(step_scores)
        data_new['stage2_pick'] = None
        if len(_flags) >= 2:
            try:
                _steps = sample['steps']
                _n = len(step_scores)
                _numbered = "\\n".join("Paragraph %d: %s" % (_i + 1, _steps[_i]) for _i in range(min(_n, len(_steps))))
                _cand = ", ".join(str(_i + 1) for _i in _flags)
                _sys = ("You are a math teacher. Below is a competition problem and its full step-by-step "
                        "solution in numbered paragraphs. These paragraphs are each SUSPECTED to contain an "
                        "error: " + _cand + ". Carefully determine which is the FIRST paragraph that contains a "
                        "genuine, concrete mathematical error. Some suspicions may be false. If, on careful "
                        "check, NONE of the suspected paragraphs actually contains an error, answer 0. "
                        "Give your final answer as \\\\boxed{N} where N is the paragraph number.")
                _usr = ("Problem: %s\\n\\n%s\\n\\nAmong paragraphs %s, which is the FIRST paragraph that contains "
                        "a genuine error? Answer with \\\\boxed{N}." % (sample['problem'], _numbered, _cand))
                _prompt = genprm.tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': _sys}, {'role': 'user', 'content': _usr}],
                    tokenize=False, add_generation_prompt=True)
                from vllm import SamplingParams as _SP
                _sp = _SP(temperature=0.6, top_p=0.95, top_k=20, max_tokens=3000)
                _o = genprm.model.generate(_prompt, _sp, use_tqdm=False)[0].outputs[0].text
                import re as _re2
                _ms = _re2.findall(r'oxed\\{?\\s*(\\d+)', _o)
                _pick = int(_ms[-1]) if _ms else None
                if _pick is not None:
                    data_new['stage2_pick'] = _pick
                    _nv = [1.0] * _n
                    if 1 <= _pick <= _n:
                        _nv[_pick - 1] = 0.0
                    step_scores = _nv
            except Exception:
                traceback.print_exc()
        # ===== end Stage 2 =====
'''
assert s.count(anchor)==1, ('anchor', s.count(anchor))
s=s.replace(anchor, block + anchor)
open(D,'w',encoding='utf-8').write(s)
print('choose variant written; ok:', ('Stage 2: 比较式复核' in s) and ('stage2_pick' in s))
