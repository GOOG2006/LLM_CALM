"""dump 失败案例:真错步的分数 + 文本 + GenPRM 对它的分析片段。诊断为什么漏/错判。
用法: python dump_failures.py <config> <out_dir> [max]"""
import sys, json, glob, os

config, outdir = sys.argv[1], sys.argv[2]
MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 8
gold = {d["id"]: d for d in json.load(open(f"/root/autodl-tmp/genprm_work/ProcessBench/{config}.json"))}

def pred(v, thr=0.5):
    for i, s in enumerate(v):
        try: s = float(s)
        except: s = 0.5
        if s < thr: return i
    return -1

rows = []
for f in glob.glob(os.path.join(outdir, "*", "result_*.json")):
    try:
        d = json.load(open(f, encoding="utf-8"))
        if "value" in d and "id" in d: rows.append(d)
    except Exception: pass

def asst_for_step(conv, k):  # step k(0-idx) 的 assistant 分析在 conv 索引 2k+2 附近
    a = [m for m in conv if m.get("role") == "assistant"]
    return a[k].get("content", "") if k < len(a) else ""

missed=[]; misloc=[]; fp=[]
for d in rows:
    L=int(d["label"]); p=pred(d["value"])
    if L>=0 and p==L: continue
    if L>=0 and p==-1: missed.append(d)
    elif L>=0: misloc.append(d)
    elif p!=-1: fp.append(d)

print(f"=== {os.path.basename(outdir)} ({config}) 失败: 漏检{len(missed)} 定位错{len(misloc)} FP{len(fp)} ===\n")
for tag, lst in [("漏检(真错步被判过关)", missed), ("定位错", misloc), ("FP(全对被判有错)", fp)]:
    for d in lst[:MAX]:
        L=int(d["label"]); g=gold.get(d["id"],{}); steps=g.get("steps",[])
        sc=[round(float(x),2) for x in d["value"]]
        print(f"[{tag}] id={d['id']} label={L} pred={pred(d['value'])} scores={sc}")
        if L>=0 and L<len(steps):
            print(f"  真错步文本: {steps[L][:180]}")
            print(f"  该步分数={sc[L] if L<len(sc) else '?'}  GenPRM分析片段: {asst_for_step(d['conversation'],L)[:260]}")
        print()
