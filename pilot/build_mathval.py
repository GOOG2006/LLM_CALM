"""建 MATH-train held-out 调参集(探针训练未用的 conv, index>=12000)。
写成 ProcessBench 格式 sample.json {problem, steps, label=首错步}。
"""
import os, re, json, numpy as np
import pyarrow.parquet as pq

N = 150
OUT = '/root/autodl-tmp/genprm_work/pb_mathval_in'
tbl = pq.read_table('/root/autodl-tmp/genprm_work/GenPRM-Data/data/train-00000-of-00001.parquet').to_pylist()
idx = np.random.default_rng(0).permutation(len(tbl))    # same seed as probe training


def parse_conv(conv):
    users = [m['content'] for m in conv if m['role'] == 'user']
    asts = [m['content'] for m in conv if m['role'] == 'assistant']
    n = min(len(users), len(asts))
    labels = [1 if (re.findall(r'boxed\{(Yes|No)\}', a)[-1:] == ['No']) else 0 for a in asts[:n]]
    u0 = re.sub(r'^\s*Question:\s*', '', users[0])
    problem, step0 = (u0.split('\n\n', 1) if '\n\n' in u0 else ('', u0))
    steps = [step0] + users[1:n]
    return problem, steps, labels[:len(steps)]


os.makedirs(OUT, exist_ok=True)
cnt = 0
for pos in range(12000, len(idx)):          # held out from probe training (first 12000)
    if cnt >= N:
        break
    p, s, lab = parse_conv(tbl[int(idx[pos])]['conversations'])
    s = [x for x in s if isinstance(x, str) and x]
    if len(s) < 2 or len(lab) != len(s):
        continue
    fe = next((i for i, v in enumerate(lab) if v == 1), -1)
    d = {'problem': p, 'steps': s, 'label': fe, 'final_answer_correct': fe == -1}
    dd = os.path.join(OUT, f'mval-{cnt}')
    os.makedirs(dd, exist_ok=True)
    json.dump(d, open(os.path.join(dd, 'sample.json'), 'w'), ensure_ascii=False)
    cnt += 1
print(f'wrote {cnt} folders to {OUT}')
