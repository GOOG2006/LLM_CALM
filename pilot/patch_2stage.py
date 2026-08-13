"""两阶段判决变体:Stage-1 照常;对被判No(reward<0.5)且未干净执行代码的步,
Stage-2 强制 verify=execute=True 重验;仅当代码执行([Code Output])且翻成Yes(reward>=0.5)才采纳翻转。
只翻'算术假阳',保留真检出/证明步。基于 prm_evaluate_adaptive5.py 生成 _2stage.py。"""
S='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D='/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5_2stage.py'
s=open(S,encoding='utf-8').read()

anchor = """            conversation[step_index] = {
                'role': 'assistant',
                'content': outputs[0]
            }
            step_scores.append(reward)
            used_code_list.append(use_code_this_step)"""

stage2 = """            # ===== Stage 2: 复核被判No的步(强制代码重验,仅代码执行且翻Yes才翻转)=====
            _CO = '[Code Output]'
            stage2_flip = False
            if reward < 0.5 and not (use_code_this_step and _CO in outputs[0]):
                out2, reward2 = genprm.inference(
                    messages=paths,
                    majority_num=1,
                    cur_step=cur_step,
                    analyze=args.analyze,
                    verify=True,
                    execute=True,
                    time_limit=3,
                    max_tokens=2048,
                    code_executor=code_executor,
                    analyze_template=args.analyze_template,
                    verify_template=args.verify_template,
                    output_template=args.output_template,
                    logging=True
                )
                if (_CO in out2[0]) and (reward2 >= 0.5):
                    outputs = out2
                    reward = reward2
                    stage2_flip = True
            # ===== end Stage 2 =====

            conversation[step_index] = {
                'role': 'assistant',
                'content': outputs[0]
            }
            step_scores.append(reward)
            used_code_list.append(use_code_this_step)
            stage2_flips.append(stage2_flip)"""

assert s.count(anchor)==1, ('anchor count', s.count(anchor))
s=s.replace(anchor, stage2)

# 初始化 stage2_flips 列表(在 used_code_list 初始化处)
init_anchor = "        used_code_list = []\n"
assert s.count(init_anchor)>=1
s=s.replace(init_anchor, "        used_code_list = []\n        stage2_flips = []\n", 1)

# 存进结果
save_anchor = "        data_new['used_code'] = used_code_list\n"
assert s.count(save_anchor)==1
s=s.replace(save_anchor, "        data_new['used_code'] = used_code_list\n        data_new['stage2_flips'] = stage2_flips\n")

open(D,'w',encoding='utf-8').write(s)
print('2stage variant written; ok:', ('Stage 2' in s) and ('stage2_flips.append' in s) and ('stage2_flips' in s))
