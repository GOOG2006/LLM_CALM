"""方案2: 探针 + 程序化算术核对(无模型)。gsm8k。
抽 'A op B = C' 重算比对, 算错=ERROR。用来给探针加召回。
探针P/t 从 probeP_gsm8k.json; 步文本/标签 从 pb_gsm8k_120_in。
"""
import json, glob, os, re, numpy as np

pp = json.load(open('probeP_gsm8k.json'))
t_abs = pp['t_abs']; probeP = pp['probeP']
DATA = 'pb_gsm8k_120_in'

NUM = r'-?\d[\d,]*\.?\d*'
PAT = re.compile('(' + NUM + r')\s*([+\-*/×·÷x])\s*(' + NUM + r')\s*=\s*(' + NUM + ')')


def arith_wrong(text):
    """返回 (发现算错, 可核验个数)。保守: 只判二元算式, 相对容差1e-4。"""
    wrong = 0; nchk = 0
    for m in PAT.finditer(text or ''):
        a, op, b, c = m.groups()
        try:
            a = float(a.replace(',', '')); b = float(b.replace(',', '')); c = float(c.replace(',', ''))
        except Exception:
            continue
        op = {'×': '*', '·': '*', '÷': '/', 'x': '*'}.get(op, op)
        if op == '/' and b == 0:
            continue
        val = {'+': a + b, '-': a - b, '*': a * b, '/': (a / b if b else None)}[op]
        if val is None:
            continue
        nchk += 1
        if abs(val - c) > 1e-4 * max(1.0, abs(c)):
            wrong += 1
    return wrong > 0, nchk


cases = []
for fo in sorted(glob.glob(os.path.join(DATA, '*'))):
    if not os.path.isdir(fo):
        continue
    nm = os.path.basename(fo)
    if nm not in probeP:
        continue
    d = json.load(open(os.path.join(fo, 'sample.json')))
    steps = [s for s in d['steps'] if s is not None]
    if steps and steps[-1] == '':
        steps = steps[:-1]
    P = probeP[nm]
    if len(P) != len(steps):
        continue
    aw = [arith_wrong(s) for s in steps]
    cases.append(dict(label=d['label'], P=np.array(P),
                      wrong=[x[0] for x in aw], nchk=[x[1] for x in aw]))
tot_chk = sum(sum(1 for n in c['nchk'] if n > 0) for c in cases)
tot_wrong = sum(sum(c['wrong']) for c in cases)
tot_steps = sum(len(c['P']) for c in cases)
print(f"cases={len(cases)} steps={tot_steps} steps_with_arith={tot_chk} steps_flagged_wrong={tot_wrong} t_abs={t_abs:.2f}", flush=True)


def f1(pred_of):
    eh = et = ch = ct = 0
    for c in cases:
        pred = pred_of(c)
        if c['label'] == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == c['label'])
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100, ae * 100, ac * 100


def probe_only(c):
    return next((i for i in range(len(c['P'])) if c['P'][i] > t_abs), -1)


def arith_only(c):
    return next((i for i in range(len(c['wrong'])) if c['wrong'][i]), -1)


def hyb_override(c):        # error if arith wrong OR probe says error
    return next((i for i in range(len(c['P'])) if c['wrong'][i] or c['P'][i] > t_abs), -1)


def hyb_gated(m):          # arith override only on probe-uncertain steps
    def f(c):
        return next((i for i in range(len(c['P']))
                     if (c['wrong'][i] if abs(c['P'][i] - t_abs) < m else False) or c['P'][i] > t_abs), -1)
    return f


print(f"\n{'policy':22s} {'F1':>6} {'accE':>6} {'accC':>6}")
for name, fn in [('probe-only', probe_only), ('arith-only', arith_only),
                 ('hybrid override', hyb_override),
                 ('hybrid gated m=.25', hyb_gated(0.25)), ('hybrid gated m=.4', hyb_gated(0.40))]:
    a, e, c = f1(fn)
    print(f"{name:22s} {a:6.1f} {e:6.1f} {c:6.1f}", flush=True)
