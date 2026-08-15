"""独立重采样: 固定用'算法已写对'的详细 skill, max_new_tokens=8000 防截断, 强制只输出完整代码。
每轮全新独立生成(不做自我纠错), 第一个能跑通并打印 SELECTED 的就停。最多 N 轮。
用法: python agentic_loop_v3.py [nrounds]
"""
import re, sys, subprocess, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'models/DSR1-7B'; PY = sys.executable
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()

SKILL = r"""# SKILL: select-readout-layer
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

## Load the arrays (COPY THESE TWO LINES EXACTLY; `allow_pickle=True` is REQUIRED because val_feats is an object array, or loading raises ValueError)
    d = np.load('layerfeats.npz', allow_pickle=True)
    cand = d['cand']; Xtr = d['Xtr']; ytr = d['ytr']; val_feats = d['val_feats']; val_fe = d['val_fe']

## Algorithm (iterate by POSITION j = 0..n_cand-1)
for j in range(len(cand)):
    scaler = StandardScaler().fit(Xtr[j])                 # Xtr[j] is [n_train_steps, H]
    clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(scaler.transform(Xtr[j]), ytr)  # fit ONCE
    preds = []
    for i in range(len(val_feats)):
        p = clf.predict_proba(scaler.transform(val_feats[i][j]))[:, 1]   # per-step prob, shape [n_steps_i]
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

## Output rules (critical)
- Keep your reasoning to at most two sentences, then output the code.
- Output exactly ONE ```python code block with a COMPLETE, runnable script (about 40 lines). Do NOT get cut off.
- The code block must contain ONLY Python. No prose, no 'Step:' sentences, no markdown inside it.
- The script must end with the `SELECTED = X` line."""


def gen():
    ids = tok.apply_chat_template([{"role": "user", "content": SKILL}], tokenize=True, add_generation_prompt=True, return_tensors='pt').cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8000, do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def extract(text):
    b = re.findall(r"```python\s*(.*?)```", text, re.DOTALL) or re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    # pick the block that is the real script (has imports + the classifier/loop), longest wins
    scriptish = [x for x in b if 'import' in x and ('LogisticRegression' in x or 'for ' in x)]
    code = max(scriptish, key=len) if scriptish else (max(b, key=len) if b else text)
    if 'import' in code:
        code = code[code.index('import'):]
    return code


log = open('loop3_transcript.txt', 'w', encoding='utf-8')
for rnd in range(1, N + 1):
    text = gen()
    code = extract(text)
    open('model_selector.py', 'w').write(code)
    try:
        r = subprocess.run([PY, 'model_selector.py'], capture_output=True, text=True, timeout=300)
        out = (r.stdout or '') + (r.stderr or ''); rc = r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or '') + (e.stderr or '') + "\nERROR: timed out 300s."; rc = -1
    sel = re.search(r"SELECTED\s*=\s*(-?\d+)", out)
    ok = (rc == 0) and (sel is not None)
    log.write(f"\n===== SAMPLE {rnd} =====\n--- CODE ---\n{code}\n--- OUTPUT ---\n{out[-1200:]}\n--- ok={ok} ---\n"); log.flush()
    print(f"SAMPLE {rnd}: rc={rc} SELECTED={sel.group(1) if sel else None} ok={ok}", flush=True)
    if ok:
        open('selected_layer.txt', 'w').write(sel.group(1))
        print(f"SUCCESS sample {rnd}: SELECTED = {sel.group(1)} (written to selected_layer.txt)", flush=True)
        break
log.close()
print("done", flush=True)
