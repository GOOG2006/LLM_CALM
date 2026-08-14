"""诊断 omnimath 为何救不回: 每域 P_A / P_B 的 step-AUC、真错步的 P_A 域内百分位、解长分布、错误位置。
判断是排序失效(P_A 噪声)还是别的。纯numpy 读 pa_4000.npz。用法: python gen7b_diag_omni.py pa_4000.npz
"""
import sys, numpy as np

D = np.load(sys.argv[1] if len(sys.argv) > 1 else 'pa_4000.npz', allow_pickle=True)
DOMS = ['gsm8k', 'math', 'olympiad', 'omnimath']


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return 0.5 if n1 == 0 or n0 == 0 else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print(f"{'dom':10s} {'ncase':>6} {'nerr':>5} {'avglen':>7} {'PA-AUC':>7} {'PB-AUC':>7} {'errPct-PA':>10} {'errPct-PB':>10}")
for d in DOMS:
    labs = D[d + '_lab']; PAs = D[d + '_PA']; PBs = D[d + '_PB']
    # step-level: build (is_error_step, PA, PB) for cases with a known error (label>=0), error step = label
    ys, sA, sB = [], [], []
    lens = []; err_pct_A, err_pct_B = [], []
    nerr = 0
    for lab, pa, pb in zip(labs, PAs, PBs):
        pa = np.asarray(pa, float); pb = np.asarray(pb, float); n = min(len(pa), len(pb)); lens.append(n)
        if lab == -1 or lab >= n:
            continue
        nerr += 1
        for i in range(n):
            ys.append(1 if i == lab else 0); sA.append(pa[i]); sB.append(1 - pb[i])
        # percentile of true-error step's score within its own case (1.0 = highest -> best)
        err_pct_A.append((pa[:n] <= pa[lab]).mean())
        err_pct_B.append(((1 - pb[:n]) <= (1 - pb[lab])).mean())
    print(f"{d:10s} {len(labs):6d} {nerr:5d} {np.mean(lens):7.1f} {auc(ys, sA):7.3f} {auc(ys, sB):7.3f} "
          f"{np.mean(err_pct_A):10.3f} {np.mean(err_pct_B):10.3f}")

# omnimath deep-dive: length-stratified PA-AUC
print("\n-- omnimath PA-AUC by solution length --")
labs = D['omnimath_lab']; PAs = D['omnimath_PA']; PBs = D['omnimath_PB']
buckets = {'<=6': [], '7-12': [], '>12': []}
for lab, pa, pb in zip(labs, PAs, PBs):
    pa = np.asarray(pa, float); n = len(pa)
    if lab == -1 or lab >= n:
        continue
    key = '<=6' if n <= 6 else ('7-12' if n <= 12 else '>12')
    for i in range(n):
        buckets[key].append((1 if i == lab else 0, pa[i]))
for k, v in buckets.items():
    if v:
        ys = [a for a, _ in v]; ss = [b for _, b in v]
        print(f"  len {k:6s}: nsteps={len(v):4d} nerrcase~{sum(ys):3d} PA-AUC={auc(ys, ss):.3f}")

# error position: early vs late
print("\n-- true-error position (fraction into solution) --")
for d in DOMS:
    fr = []
    for lab, pa in zip(D[d + '_lab'], D[d + '_PA']):
        n = len(np.asarray(pa));
        if lab != -1 and lab < n:
            fr.append(lab / max(n - 1, 1))
    print(f"  {d:10s} mean-pos={np.mean(fr):.2f}")
