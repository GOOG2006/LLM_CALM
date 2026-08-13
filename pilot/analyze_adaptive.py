"""分析 adaptive2 输出:代码使用率、退避触发率、按是否退避拆 F1。诊断为什么某子集差。
用法: python analyze_adaptive.py <out_dir>"""
import sys, json, glob, os

outdir = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(outdir, "*", "result_*.json")):
    try:
        d = json.load(open(f, encoding="utf-8"))
        if "value" in d and "label" in d:
            rows.append(d)
    except Exception:
        pass

def pred(v, thr=0.5):
    for i, s in enumerate(v):
        try: s = float(s)
        except: s = 0.5
        if s < thr: return i
    return -1

def conv_text(d):
    return " ".join((m.get("content","") if isinstance(m.get("content"),str) else "") for m in d.get("conversation",[]))

def f1(rs):
    err=[r for r in rs if int(r["label"])!=-1]; cor=[r for r in rs if int(r["label"])==-1]
    if not err or not cor: return float("nan"), len(err), len(cor)
    rec=sum(pred(r["value"])==int(r["label"]) for r in err)/len(err)
    ca=sum(pred(r["value"])==-1 for r in cor)/len(cor)
    return (2*rec*ca/max(rec+ca,1e-9), len(err), len(cor))

# 每题:代码使用率 + 是否发生退避(代码报错)
all_uc=[u for r in rows for u in r.get("used_code",[])]
code_frac = sum(all_uc)/max(len(all_uc),1)
backoff = [r for r in rows if ("Code execute Error" in conv_text(r) or "Code format error" in conv_text(r))]
# 退避 = 用了代码 且 代码报过错(触发一击退避)
used_any_code = [r for r in rows if any(r.get("used_code",[]))]

F,ne,nc = f1(rows)
print(f"{os.path.basename(outdir)}: {len(rows)}案 (err{ne}/cor{nc})  F1={F*100:.1f}")
print(f"  步级代码使用率={code_frac*100:.0f}%  用过代码的题={len(used_any_code)}/{len(rows)}")
print(f"  代码报错(触发退避)的题={len(backoff)}/{len(rows)} = {100*len(backoff)/max(len(rows),1):.0f}%")

# 按"是否触发退避"拆 F1
noback = [r for r in rows if r not in backoff]
Fb,_,_ = f1(backoff); Fn,_,_ = f1(noback)
print(f"  触发退避的题 F1={Fb*100:.1f} (n={len(backoff)})  |  未退避 F1={Fn*100:.1f} (n={len(noback)})")
