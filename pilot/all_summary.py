"""4 子集 x 各方法 F1 汇总 + 同案例匹配(no-code vs adaptive)。纯读文件。"""
import json, glob, os

def load(dd):
    m = {}
    for f in glob.glob(os.path.join(dd, "*", "result_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
            if "value" in d and "id" in d: m[d["id"]] = d
        except Exception: pass
    return m

def pred(v):
    for i, s in enumerate(v):
        try: s = float(s)
        except: s = 0.5
        if s < 0.5: return i
    return -1

def f1(dmap, ids=None):
    rows = [dmap[i] for i in (ids or dmap) if i in dmap]
    err = [r for r in rows if int(r["label"]) != -1]; cor = [r for r in rows if int(r["label"]) == -1]
    if not err or not cor: return None, len(err), len(cor)
    rec = sum(pred(r["value"]) == int(r["label"]) for r in err)/len(err)
    ca  = sum(pred(r["value"]) == -1 for r in cor)/len(cor)
    return 2*rec*ca/max(rec+ca,1e-9)*100, len(err), len(cor)

def cr(dmap):
    uc=[u for d in dmap.values() for u in d.get("used_code",[])]
    return 100*sum(uc)/len(uc) if uc else None

subs = ["gsm8k","math","olympiadbench","omnimath"]
# method -> per-sub dir template (list of dirs to merge)
METH = {
 "no-code":      lambda s:[f"pb_{s}_ao_out"],
 "adaptive2":    lambda s:[f"pb_{s}_n20_a2_out", f"pb_{s}_add_a2_out"],
 "adaptive5":    lambda s:[f"pb_{s}_n20_a5_out", f"pb_{s}_add_a5_out"],
 "full(with code)": lambda s:{"gsm8k":["pb_gsm8k_full_out"],"math":["pb_math_repro_out"]}.get(s,[]),
}
def mload(dirs):
    m={}
    for d in dirs:
        if os.path.isdir(d): m.update(load(d))
    return m

print("=== 各方法 各子集 F1(自身案例)===")
hdr=f"{'method':<18}"+"".join(f"{s:>16}" for s in subs); print(hdr)
data={}
for mname,fn in METH.items():
    cells=[]; data[mname]={}
    for s in subs:
        m=mload(fn(s)); data[mname][s]=m
        F,ne,nc=f1(m)
        cells.append(f"{F:.1f}({ne}+{nc})" if F is not None else "-")
    print(f"{mname:<18}"+"".join(f"{c:>16}" for c in cells))

print("\n=== 同案例匹配:no-code vs adaptive2 vs adaptive5(每子集共同id)===")
print(f"{'subset':<16}{'n':>5}{'no-code':>10}{'adaptive2':>11}{'adaptive5':>11}")
for s in subs:
    nc=data["no-code"][s]; a2=data["adaptive2"][s]; a5=data["adaptive5"][s]
    common=set(nc)&set(a2)&set(a5)
    if not common:
        print(f"{s:<16}{'-':>5}"); continue
    def g(m):
        F,_,_=f1(m,common); return f"{F:.1f}" if F is not None else "-"
    print(f"{s:<16}{len(common):>5}{g(nc):>10}{g(a2):>11}{g(a5):>11}")

print("\n=== 代码率(adaptive)===")
for s in subs:
    a2=cr(data['adaptive2'][s]); a5=cr(data['adaptive5'][s])
    print(f"  {s:<14} adaptive2={a2:.0f}%  adaptive5={a5:.0f}%" if a2 else f"  {s}: -")
