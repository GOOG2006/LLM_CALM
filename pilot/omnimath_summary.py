"""汇总 OmniMATH 所有已跑结果:各方法 F1 + 代码率,并做同案例(case-id)匹配对比。纯读文件,无GPU。"""
import json, glob, os

def load(dd):
    m = {}
    for f in glob.glob(os.path.join(dd, "*", "result_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
            if "value" in d and "id" in d: m[d["id"]] = d
        except Exception: pass
    return m

def pred(v, thr=0.5):
    for i, s in enumerate(v):
        try: s = float(s)
        except: s = 0.5
        if s < thr: return i
    return -1

def f1(dmap, ids=None):
    rows = [dmap[i] for i in (ids or dmap)] if isinstance(dmap, dict) else dmap
    rows = [r for r in rows if r]
    err = [r for r in rows if int(r["label"]) != -1]; cor = [r for r in rows if int(r["label"]) == -1]
    if not err or not cor: return None, len(err), len(cor)
    rec = sum(pred(r["value"]) == int(r["label"]) for r in err) / len(err)
    ca = sum(pred(r["value"]) == -1 for r in cor) / len(cor)
    return 2*rec*ca/max(rec+ca,1e-9)*100, len(err), len(cor)

def coderate(dmap):
    uc = [u for d in dmap.values() for u in d.get("used_code", [])]
    return 100*sum(uc)/len(uc) if uc else None

methods = {
    "no-code(analyze-only)": "pb_omnimath_ao_out",
    "adaptive2 n20": "pb_omnimath_n20_a2_out",
    "adaptive2 add": "pb_omnimath_add_a2_out",
    "adaptive5 n20": "pb_omnimath_n20_a5_out",
    "adaptive5 add": "pb_omnimath_add_a5_out",
}
loaded = {k: load(v) for k, v in methods.items() if os.path.isdir(v)}

print("=== OmniMATH 各方法(自身全部案例)===")
for k, m in loaded.items():
    F, ne, nc = f1(m); cr = coderate(m)
    fs = f"{F:.1f}" if F is not None else "NA"
    cs = f"{cr:.0f}%" if cr is not None else "0%(无代码)"
    print(f"  {k:<24} F1={fs}  (err{ne}/cor{nc})  代码率={cs}")

# 合并 adaptive2/5 的 n20+add = n40
def merge(a, b): m = dict(a); m.update(b); return m
if "adaptive2 n20" in loaded and "adaptive2 add" in loaded:
    a2_40 = merge(loaded["adaptive2 n20"], loaded["adaptive2 add"])
    a5_40 = merge(loaded["adaptive5 n20"], loaded["adaptive5 add"])
    print("\n=== 合并 n=40 ===")
    for k, m in [("adaptive2 n40", a2_40), ("adaptive5 n40", a5_40)]:
        F, ne, nc = f1(m); print(f"  {k:<16} F1={F:.1f} (err{ne}/cor{nc}) 代码率={coderate(m):.0f}%")

    # 同案例匹配:三方共同 id
    common = set(a2_40) & set(a5_40) & set(loaded.get("no-code(analyze-only)", {}))
    print(f"\n=== 同案例匹配对比(共同 {len(common)} 案)===")
    for k, m in [("no-code", loaded["no-code(analyze-only)"]), ("adaptive2", a2_40), ("adaptive5", a5_40)]:
        F, ne, nc = f1(m, common)
        print(f"  {k:<12} F1={F:.1f} (err{ne}/cor{nc})" if F else f"  {k}: 匹配案例不足")
