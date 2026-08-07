"""在正确工作的 GenPRM 官方输出上重验缺陷 W1/W3/W4/W2 + 定位失败靶点。
用法: python analyze_good_genprm.py <out_dir>"""
import sys, json, glob, os
from collections import Counter

outdir = sys.argv[1]
files = glob.glob(os.path.join(outdir, "*", "result_*.json"))
rows = []
for f in files:
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
    c = d.get("conversation", [])
    return " ".join(m.get("content","") if isinstance(m.get("content"), str) else str(m.get("content","")) for m in c)

err = [r for r in rows if int(r["label"]) != -1]
cor = [r for r in rows if int(r["label"]) == -1]
allsc = [float(s) for r in rows for s in r["value"]]
print(f"案例 {len(rows)} (err {len(err)}/cor {len(cor)})  总步 {len(allsc)}\n")

# W1 饱和
sat = sum(1 for s in allsc if s>=0.9)/max(len(allsc),1)
half = sum(1 for s in allsc if abs(s-0.5)<1e-6)/max(len(allsc),1)
print(f"[W1 饱和] 步分数均值={sum(allsc)/len(allsc):.3f}  >=0.9占比={sat*100:.0f}%  ==0.5占比(W5)={half*100:.0f}%")

# W3/W4 代码报错
ncode_err = sum(conv_text(r).count("Code execute Error") + conv_text(r).count("Code format error") for r in rows)
cases_with_err = sum(1 for r in rows if ("Code execute Error" in conv_text(r) or "Code format error" in conv_text(r)))
print(f"[W3/W4 代码] 代码报错总次数={ncode_err}  出现报错的案例={cases_with_err}/{len(rows)}")

# 定位失败靶点(err案例)
hit=misloc=missed=0
for r in err:
    p=pred(r["value"]); L=int(r["label"])
    if p==L: hit+=1
    elif p==-1: missed+=1
    else: misloc+=1
fp=sum(1 for r in cor if pred(r["value"])!=-1)
print(f"\n[定位] err案例: 命中={hit} 定位错={misloc} 完全漏检={missed} (共{len(err)})")
print(f"[FP] cor案例误报={fp}/{len(cor)}")

# W2 早承诺: 定位错的案例里,预测步 vs 真首错步
early=sum(1 for r in err if pred(r['value'])!=-1 and pred(r['value'])<int(r['label']))
late=sum(1 for r in err if pred(r['value'])!=-1 and pred(r['value'])>int(r['label']))
print(f"[W2 早/晚承诺] 定位错中: 早于真错={early} 晚于真错={late}")

# 链长 vs 命中(W2)
def bin_hit(lo,hi):
    sub=[r for r in err if lo<=len(r['value'])<=hi]
    if not sub: return None
    return len(sub), sum(pred(r['value'])==int(r['label']) for r in sub)/len(sub)
print("[链长 vs err命中率]")
for lo,hi,nm in [(1,6,'短'),(7,12,'中'),(13,99,'长')]:
    b=bin_hit(lo,hi)
    if b: print(f"  {nm}({lo}-{hi}步): n={b[0]} 命中率={b[1]:.2f}")
