"""1.5B standalone: 读 pa15_full.npz, 对比域自适应策略 A/B/C 对 fthead-1.5B 定位F1(全量 ProcessBench)。
阈值在 math-val 调, 校准只用测试域自身分布(无标签)。对比 GenPRM-1.5B 论文值 + fthead 基线58.1。
用法: python gen15_calib.py pa15_full.npz
"""
import sys, numpy as np

D = np.load(sys.argv[1] if len(sys.argv) > 1 else 'pa15_full.npz', allow_pickle=True)
DOMS = ['gsm8k', 'math', 'olympiad', 'omnimath']
GENPRM = {'gsm8k': 52.8, 'math': 66.6, 'olympiad': 55.1, 'omnimath': 54.5}   # 论文全量
BASE = {'gsm8k': 65.1, 'math': 61.3, 'olympiad': 53.5, 'omnimath': 52.4}     # fthead 全局阈值(58.1均值)


def lz(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def locf1(cases, thr, evfn):
    eh = et = ch = ct = 0
    for lab, pa in cases:
        ev = evfn(pa); pred = next((i for i in range(len(ev)) if ev[i] > thr), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); return 2 * ae * ac / max(ae + ac, 1e-9) * 100


cases = {d: list(zip(D[d + '_lab'], D[d + '_PA'])) for d in DOMS}
val = list(zip(D['val_fe'], D['val_PA']))


def pooled(cl):
    return np.concatenate([lz(pa) for _, pa in cl])


def make_ev(kind, cl):
    e = pooled(cl)
    if kind == 'A':
        m, s = 0.0, 1.0
    elif kind == 'B':
        m, s = np.median(e), 1.0
    elif kind == 'C':
        m, s = e.mean(), e.std() + 1e-6
    return lambda pa: (lz(pa) - m) / s


def tune_thr(kind):
    ev = make_ev(kind, val); bt, bf = 0.0, -1
    lo, hi = (-3, 3) if kind != 'A' else (-4, 4)
    for t in np.arange(lo, hi, 0.1):
        f = locf1(val, t, ev)
        if f > bf:
            bf, bt = f, t
    return bt


print(f"{'strategy':12s} " + " ".join(f"{d:>9}" for d in DOMS) + f"{'MEAN':>9}")
for kind in ['A', 'B', 'C']:
    thr = tune_thr(kind); fs = []
    for d in DOMS:
        fs.append(locf1(cases[d], thr, make_ev(kind, cases[d])))
    print(f"{kind} thr={thr:5.2f}  " + " ".join(f"{x:9.1f}" for x in fs) + f"{np.mean(fs):9.1f}")
print(f"{'fthead-base':12s} " + " ".join(f"{BASE[d]:9.1f}" for d in DOMS) + f"{np.mean(list(BASE.values())):9.1f}")
print(f"{'GenPRM-1.5B':12s} " + " ".join(f"{GENPRM[d]:9.1f}" for d in DOMS) + f"{np.mean(list(GENPRM.values())):9.1f}")
