"""参照选层器(标准答案): 按规范用 logreg + locF1 选层。核对基座生成代码是否一致。"""
import numpy as np
from sklearn.linear_model import LogisticRegression
d = np.load('layerfeats.npz', allow_pickle=True)
cand = d['cand']; Xtr = d['Xtr']; ytr = d['ytr']; vf = d['val_feats']; fe = d['val_fe']
best = (-1, None)
for j, L in enumerate(cand):
    mu = Xtr[j].mean(0); sd = Xtr[j].std(0) + 1e-6
    clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit((Xtr[j] - mu) / sd, ytr)
    eh = et = ch = ct = 0
    for i in range(len(vf)):
        P = clf.predict_proba((vf[i][j] - mu) / sd)[:, 1]
        pred = next((k for k in range(len(P)) if P[k] > 0.5), -1)
        if fe[i] == -1:
            ct += 1; ch += (pred == -1)
        else:
            et += 1; eh += (pred == fe[i])
    ae = eh / max(et, 1); ac = ch / max(ct, 1); f1 = 2 * ae * ac / max(ae + ac, 1e-9) * 100
    print(f"layer {int(L):4d}  F1={f1:.1f}")
    if f1 > best[0]:
        best = (f1, int(L))
print(f"SELECTED = {best[1]}  (F1={best[0]:.1f})")
