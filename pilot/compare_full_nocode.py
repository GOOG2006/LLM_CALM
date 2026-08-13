"""对比满配(带代码)vs 去代码,同案例。找"代码救了场"的案例并 dump 代码内容,分析为什么代码在该子集有用。
用法: python compare_full_nocode.py <config> <full_dir> <nocode_dir> [maxdump]"""
import sys, re, json, glob, os
config, fulld, nod = sys.argv[1], sys.argv[2], sys.argv[3]
MAX = int(sys.argv[4]) if len(sys.argv) > 4 else 6
gold = {d["id"]: d for d in json.load(open(f"/root/autodl-tmp/genprm_work/ProcessBench/{config}.json"))}

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
def asst(conv): return [m.get("content","") for m in conv if m.get("role")=="assistant"]

full, noc = load(fulld), load(nod)
ids = sorted(set(full) & set(noc))
def correct(d): return pred(d["value"]) == int(d["label"])
helps = [i for i in ids if correct(full[i]) and not correct(noc[i])]
hurts = [i for i in ids if correct(noc[i]) and not correct(full[i])]
print(f"=== {config}: 共同案例 {len(ids)} | 代码救场(满配对/去代码错)={len(helps)} | 代码帮倒忙={len(hurts)} ===\n")

for i in helps[:MAX]:
    L = int(full[i]["label"]); g = gold.get(i, {}); steps = g.get("steps", [])
    fv = [round(float(x),2) for x in full[i]["value"]]; nv = [round(float(x),2) for x in noc[i]["value"]]
    print(f"[代码救场] id={i} label={L}")
    print(f"  满配scores ={fv}  (pred={pred(full[i]['value'])})")
    print(f"  去代码scores={nv}  (pred={pred(noc[i]['value'])})")
    if 0 <= L < len(steps):
        print(f"  真错步: {steps[L][:150]}")
        A = asst(full[i]["conversation"])
        t = A[L] if L < len(A) else ""
        code = re.search(r'```python(.*?)```', t, re.DOTALL)
        cout = re.search(r'\[Code Output\](.*?)```(.*?)```', t, re.DOTALL)
        print(f"  满配对该步的代码: {(code.group(1).strip()[:200]) if code else '(无代码)'}")
        print(f"  代码输出: {(cout.group(2).strip()[:120]) if cout else '(无)'}")
    print()
