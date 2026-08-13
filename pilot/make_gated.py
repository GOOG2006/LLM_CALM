"""生成 prm_evaluate_gated.py = adaptive5 但 route 改为读探针门控 gate json。"""
S = '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_adaptive5.py'
D = '/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/prm_evaluate_gated.py'
s = open(S, encoding='utf-8').read()

anchor = "version = 'v2.3'"
inject = anchor + ("\nimport json as _json, os as _os\n"
                   "_GATE = _json.load(open(_os.environ.get('GATE_FILE', 'gate_gsm8k.json')))\n"
                   "def _use_gate(folder, step):\n"
                   "    g = _GATE.get(folder, [])\n"
                   "    return bool(g[step-1]) if 0 <= step-1 < len(g) else False")
assert s.count(anchor) == 1, ('anchor', s.count(anchor))
s = s.replace(anchor, inject)

old = "use_code_this_step = code_enabled and route_use_code(conversation[step_index-1].get('content','') if step_index>0 else '')"
new = "use_code_this_step = code_enabled and _use_gate(folder_name, cur_step)"
assert s.count(old) == 1, ('use_code line', s.count(old))
s = s.replace(old, new)

open(D, 'w', encoding='utf-8').write(s)
import ast; ast.parse(s)
print('wrote prm_evaluate_gated.py, syntax OK')
