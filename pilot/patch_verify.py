"""改进版 Stage-2(保守逐个核验):Stage-1 照常;若级联(>=2 flag),
从第一个被判No的步开始,逐个问'这段是不是真错?(\\boxed{Yes/No}, Yes=有错)',
第一个被确认为真错的 flag 作为预测;被判为正确的才前移;全部否定则判全对(-1)。
默认保留第一个 flag(除非被明确否定)。基于 prm_evaluate_adaptive5.py 生成 _verify.py。"""
S='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5_verify.py'
s=open(S,encoding='utf-8').read()

anchor = "        data_new['value'] = step_scores\n"
block = '''        # ===== Stage 2: 保守逐个核验(仅级联 >=2 flag)=====
        _flags = [ _i for _i, _s in enumerate(step_scores) if float(_s) < 0.5 ]
        data_new['value_stage1'] = list(step_scores)
        data_new['stage2_conf'] = None
        if len(_flags) >= 2:
            try:
                _steps = sample['steps']
                _n = len(step_scores)
                _numbered = "\\n".join("Paragraph %d: %s" % (_i + 1, _steps[_i]) for _i in range(min(_n, len(_steps))))
                from vllm import SamplingParams as _SP
                _sp = _SP(temperature=0.6, top_p=0.95, top_k=20, max_tokens=3000)
                import re as _re2
                _confirmed = None
                for _fi in _flags[:6]:
                    _N = _fi + 1
                    _sys = ("You are a math teacher. Below is a competition problem and its full step-by-step "
                            "solution in numbered paragraphs. Focus ONLY on Paragraph %d. With the full solution "
                            "visible for context, decide whether Paragraph %d itself contains a genuine, concrete "
                            "mathematical error (a wrong calculation, a false statement, or an invalid deduction), "
                            "or whether Paragraph %d is actually correct. Answer \\\\boxed{Yes} if Paragraph %d "
                            "genuinely contains an error, or \\\\boxed{No} if it is actually correct." % (_N, _N, _N, _N))
                    _usr = ("Problem: %s\\n\\n%s\\n\\nDoes Paragraph %d contain a genuine error? Answer \\\\boxed{Yes} "
                            "or \\\\boxed{No}." % (sample['problem'], _numbered, _N))
                    _prompt = genprm.tokenizer.apply_chat_template(
                        [{'role': 'system', 'content': _sys}, {'role': 'user', 'content': _usr}],
                        tokenize=False, add_generation_prompt=True)
                    _o = genprm.model.generate(_prompt, _sp, use_tqdm=False)[0].outputs[0].text
                    _ms = _re2.findall(r'oxed\\{?\\s*(Yes|No)', _o, _re2.I)
                    _ans = _ms[-1].lower() if _ms else 'yes'  # 取不到默认保留(视为真错)
                    if _ans.startswith('yes'):
                        _confirmed = _fi
                        break
                # 覆盖预测
                data_new['stage2_conf'] = _confirmed
                _nv = [1.0] * _n
                if _confirmed is not None:
                    _nv[_confirmed] = 0.0
                step_scores = _nv
            except Exception:
                traceback.print_exc()
        # ===== end Stage 2 =====
'''
assert s.count(anchor)==1, ('anchor', s.count(anchor))
s=s.replace(anchor, block + anchor)
open(D,'w',encoding='utf-8').write(s)
print('verify variant written; ok:', ('Stage 2: 保守逐个核验' in s) and ('stage2_conf' in s))
