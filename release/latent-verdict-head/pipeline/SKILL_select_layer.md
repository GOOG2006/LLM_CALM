# SKILL: select-readout-layer

This is the instruction given to the base model in Stage 2. The base model reads it and writes a
Python script that chooses the readout layer for the residual head. The script is then executed
(Stage 3). The same text lives inline in `stage2_generate_selector.py`.

Goal: from `layerfeats.npz`, choose the transformer layer that best locates the FIRST incorrect
reasoning step, and print `SELECTED = <layer label>`.

## Data (np.load('layerfeats.npz', allow_pickle=True))
- `cand`      : int array [n_cand]. Layer LABELS (e.g. [-1,-3,-5,-7,-9,-11,-13,-15]). For PRINTING only, NOT array indices.
- `Xtr`       : float array [n_cand, n_train_steps, H].
- `ytr`       : int array [n_train_steps]. 1 = error step, 0 = correct.
- `val_feats` : object array [n_val]; `val_feats[i]` is a float array [n_cand, n_steps_i, H].
- `val_fe`    : int array [n_val]; first error step index of case i, or -1 if all correct.

## Imports
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

## Load the arrays (COPY THESE TWO LINES EXACTLY; `allow_pickle=True` is REQUIRED because val_feats is an object array)
    d = np.load('layerfeats.npz', allow_pickle=True)
    cand = d['cand']; Xtr = d['Xtr']; ytr = d['ytr']; val_feats = d['val_feats']; val_fe = d['val_fe']

## Algorithm (iterate by POSITION j = 0..n_cand-1)
for j in range(len(cand)):
    scaler = StandardScaler().fit(Xtr[j])
    clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(scaler.transform(Xtr[j]), ytr)  # fit ONCE
    preds = []
    for i in range(len(val_feats)):
        p = clf.predict_proba(scaler.transform(val_feats[i][j]))[:, 1]      # per-step error prob
        preds.append(next((k for k in range(len(p)) if p[k] > 0.5), -1))
    et = sum(val_fe[i] != -1 for i in range(len(val_fe)))
    eh = sum(val_fe[i] != -1 and preds[i] == val_fe[i] for i in range(len(val_fe)))
    ct = sum(val_fe[i] == -1 for i in range(len(val_fe)))
    ch = sum(val_fe[i] == -1 and preds[i] == -1 for i in range(len(val_fe)))
    accE = eh/et; accC = ch/ct; F1 = 2*accE*accC/(accE+accC)
    print(cand[j], F1)
Then print exactly:  print(f"SELECTED = {best_label}")   where best_label is cand[j] with the highest F1.

## Pitfalls
- Never index Xtr or val_feats[i] with a layer LABEL (e.g. Xtr[-7]); always use the POSITION j.
- Never reshape features to (1,-1); predict_proba takes the [n_steps, H] matrix and returns one row per step.
- Fit each LogisticRegression exactly once per layer.

## Output rules
- Keep reasoning to at most two sentences, then output the code.
- Output exactly one Python code block with a COMPLETE runnable script that ends with the `SELECTED = X` line.
- The code block must contain only Python, no prose.
