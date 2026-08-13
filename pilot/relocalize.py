"""改进:回溯重定位。GenPRM 常在下游步才标错,但其分析文本会归因到更早步。
当某步被标错时,若其分析在错误语境下提到更早 paragraph,则把预测回溯到该步。
纯后处理现有输出,测 err-acc / F1 提升。用法: python relocalize.py <config> <out_dir...>"""
import sys, re, json, glob, os

config = sys.argv[1]
dirs = sys.argv[2:]
PARA = re.compile(r'[Pp]aragraph\s+(\d+)')
ERR = ('error', 'incorrect', 'wrong', 'mistake', 'originat', 'prior', 'earlier', 'previous', 'stems from', 'due to')

rows = []
for dd in dirs:
    for f in glob.glob(os.path.join(dd, "*", "result_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
            if "value" in d and "label" in d: rows.append(d)
        except Exception: pass

def base_pred(v, thr=0.5):
    for i, s in enumerate(v):
        try: s = float(s)
        except: s = 0.5
        if s < thr: return i
    return -1

def asst(conv):
    return [m.get("content", "") for m in conv if m.get("role") == "assistant"]

def reloc_pred(d, thr=0.5):
    p = base_pred(d["value"], thr)
    if p <= 0: return p  # 没标错 或 已是第0步,无法回溯
    A = asst(d.get("conversation", []))
    t = A[p] if p < len(A) else ""
    if not t: return p
    refs = [int(x) - 1 for x in PARA.findall(t)]           # 0-indexed
    earlier = [r for r in refs if 0 <= r < p]
    if earlier and any(w in t.lower() for w in ERR):
        return min(earlier)                                 # 回溯到它归因的最早步
    return p

def f1(rows, predfn):
    err = [r for r in rows if int(r["label"]) != -1]
    cor = [r for r in rows if int(r["label"]) == -1]
    if not err or not cor: return None
    rec = sum(predfn(r) == int(r["label"]) for r in err) / len(err)
    ca = sum(predfn(r) == -1 for r in cor) / len(cor)
    return rec, ca, 2 * rec * ca / max(rec + ca, 1e-9)

b = f1(rows, lambda r: base_pred(r["value"]))
n = f1(rows, lambda r: reloc_pred(r))
print(f"{config} ({len(rows)}案)")
print(f"  baseline:    err-acc={b[0]:.3f} cor-acc={b[1]:.3f} F1={b[2]*100:.1f}")
print(f"  回溯重定位:  err-acc={n[0]:.3f} cor-acc={n[1]:.3f} F1={n[2]*100:.1f}   ΔF1={100*(n[2]-b[2]):+.1f}")
