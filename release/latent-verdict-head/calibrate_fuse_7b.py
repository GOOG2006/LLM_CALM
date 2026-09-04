"""纯numpy: 读 pa_4000.npz, 对比域自适应阈值策略对 fthead 定位F1 的作用(尤其证明域)。
A 全局阈值 / B 域内中心化 / C 域内z-score。阈值在 math-val 调, 校准只用测试域自身分布(无标签)。
再看校准后 P_A 的诚实融合(留一域外)。用法: python gen7b_calib.py pa_4000.npz
"""
import sys, numpy as np

D = np.load(sys.argv[1] if len(sys.argv) > 1 else 'pa_4000.npz', allow_pickle=True)
DOMS = ['gsm8k', 'math', 'olympiad', 'omnimath']
REF = {'gsm8k': 83.4, 'math': 80.0, 'olympiad': 72.3, 'omnimath': 71.5}


def lz(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def locf1(cases, thr, evfn):
    """cases: list of (label, PA_array). evfn(PA)->evidence; pred=first step ev>thr."""
    eh = et = ch = ct = 0
    for lab, pa in cases:
        ev = evfn(pa); pred = next((i for i in range(len(ev)) if ev[i] > thr), -1)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); return 2 * ae * ac / max(ae + ac, 1e-9) * 100


# build per-domain case lists
cases = {d: list(zip(D[d + '_lab'], D[d + '_PA'])) for d in DOMS}
val = list(zip(D['val_fe'], D['val_PA']))


def pooled_logit(caselist):
    return np.concatenate([lz(pa) for _, pa in caselist])


# ---- strategy transforms: return (evfn_factory) that centers per its own pool ----
def make_ev(kind, caselist):
    e = pooled_logit(caselist)
    if kind == 'A':      # raw logit
        m, s = 0.0, 1.0
    elif kind == 'B':    # center by median
        m, s = np.median(e), 1.0
    elif kind == 'C':    # z-score
        m, s = e.mean(), e.std() + 1e-6
    return lambda pa: (lz(pa) - m) / s


def tune_thr(kind):
    ev = make_ev(kind, val)
    bt, bf = 0.0, -1
    lo, hi = (-3, 3) if kind != 'A' else (-4, 4)
    for t in np.arange(lo, hi, 0.1):
        f = locf1(val, t, ev)
        if f > bf:
            bf, bt = f, t
    return bt


print(f"{'strategy':10s} " + " ".join(f"{d:>9}" for d in DOMS) + f"{'MEAN':>9}")
for kind in ['A', 'B', 'C']:
    thr = tune_thr(kind)
    fs = []
    for d in DOMS:
        ev = make_ev(kind, cases[d])   # calibrate using THIS domain's own pool (unsupervised)
        fs.append(locf1(cases[d], thr, ev))
    print(f"{kind} thr={thr:5.2f} " + " ".join(f"{x:9.1f}" for x in fs) + f"{np.mean(fs):9.1f}")
print(f"{'GenPRM':10s} " + " ".join(f"{REF[d]:9.1f}" for d in DOMS) + f"{np.mean(list(REF.values())):9.1f}")

# ---- honest fusion with best calibration (leave-one-domain-out) ----
print("\n-- honest leave-one-domain-out fusion (P_A calibrated per strategy) --")
GRID = [(a, b) for a in np.arange(0, 1.01, 0.1) for b in np.arange(-3, 3.01, 0.25)]


def fuse_pred(pa, pb, evfn, a, bias):
    n = min(len(pa), len(pb)); eA = evfn(pa)[:n]; eB = lz(1 - pb[:n])
    ev = a * eA + (1 - a) * eB
    return next((i for i in range(n) if ev[i] > bias), -1)


def fuse_f1(dom_cases, evfn, a, bias):
    eh = et = ch = ct = 0
    for lab, pa, pb in dom_cases:
        pred = fuse_pred(pa, pb, evfn, a, bias)
        if lab == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == lab)
    ae = eh / max(et, 1); ac = ch / max(ct, 1); return 2 * ae * ac / max(ae + ac, 1e-9) * 100


fcases = {d: list(zip(D[d + '_lab'], D[d + '_PA'], D[d + '_PB'])) for d in DOMS}
for kind in ['A', 'B', 'C']:
    hon = []
    for d in DOMS:
        others = [(lab, pa, pb) for d2 in DOMS if d2 != d for (lab, pa, pb) in fcases[d2]]
        evf_other = make_ev(kind, [(l, pa) for (l, pa, pb) in others])
        best_ab, best_f = (0.5, 0.0), -1
        for a, b in GRID:
            f = fuse_f1(others, evf_other, a, b)
            if f > best_f:
                best_f, best_ab = f, (a, b)
        evf_d = make_ev(kind, [(l, pa) for (l, pa, pb) in fcases[d]])
        hon.append(fuse_f1(fcases[d], evf_d, *best_ab))
    print(f"{kind}: " + " ".join(f"{d}={x:.1f}" for d, x in zip(DOMS, hon)) + f"  MEAN={np.mean(hon):.1f} (GenPRM {np.mean(list(REF.values())):.1f})")
