"""组合(gsm8k): 探针打底 + 门控步(|P-t|<m)用基座+代码reward。扫多个 m。
探针P/t 从 probeP_gsm8k.json; 基座代码reward 从 pb_gsm8k_gated_out; 标签从同处 result_1.json。
"""
import json, glob, os, numpy as np

pp = json.load(open('probeP_gsm8k.json'))
t_abs = pp['t_abs']; probeP = pp['probeP']
GATED = 'pb_gsm8k_gated_out'

cases = []
for fo in sorted(glob.glob(os.path.join(GATED, '*_analyze'))):
    nm = os.path.basename(fo).replace('_analyze', '')
    r = json.load(open(os.path.join(fo, 'result_1.json')))
    if nm not in probeP:
        continue
    P = probeP[nm]; bval = r['value']
    if len(P) != len(bval):
        continue
    cases.append(dict(label=r['label'], P=np.array(P), bval=np.array(bval)))
print(f"aligned cases={len(cases)}  t_abs={t_abs:.2f}", flush=True)


def f1(pred_of):
    eh = et = ch = ct = 0; used = tot = 0
    for c in cases:
        pred, nc = pred_of(c)
        used += nc; tot += len(c['P'])
        if c['label'] == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == c['label'])
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return 2 * ae * ac / max(ae + ac, 1e-9) * 100, used / max(tot, 1) * 100


def probe_only(c):
    return next((i for i in range(len(c['P'])) if c['P'][i] > t_abs), -1), 0


def basecode_gated(c):     # base code on all steps we have (gated at m=0.4)
    return next((i for i in range(len(c['bval'])) if c['bval'][i] < 0.5), -1), len(c['bval'])


def hybrid(m):
    def f(c):
        nc = 0; err = []
        for i in range(len(c['P'])):
            if abs(c['P'][i] - t_abs) < m:
                nc += 1; err.append(c['bval'][i] < 0.5)
            else:
                err.append(c['P'][i] > t_abs)
        return next((i for i, e in enumerate(err) if e), -1), nc
    return f


print(f"\n{'policy':26s} {'F1':>6} {'code%':>6}")
print("%-26s %6.1f %6.1f" % ('probe-only', *f1(probe_only)))
print("%-26s %6.1f %6.1f" % ('basecode-only(m=.4 set)', *f1(basecode_gated)), flush=True)
for m in [0.15, 0.25, 0.35, 0.40]:
    print("%-26s %6.1f %6.1f" % (f'hybrid BASE m={m}', *f1(hybrid(m))), flush=True)
