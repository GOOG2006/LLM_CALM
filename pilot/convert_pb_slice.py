"""转换 err[a:b]+cor[a:b] 案例为官方文件夹,按 case-id 命名(避免与 n=20 序号命名冲突)。
用法: python convert_pb_slice.py <config> <a> <b> <out_dir>"""
import json, os, sys
config, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
data = json.load(open(f"/root/autodl-tmp/genprm_work/ProcessBench/{config}.json"))
err = [d for d in data if d["label"] != -1]
cor = [d for d in data if d["label"] == -1]
sub = err[a:b] + cor[a:b]
os.makedirs(out, exist_ok=True)
for d in sub:
    name = str(d["id"]).replace("/", "_")
    dd = os.path.join(out, name)
    os.makedirs(dd, exist_ok=True)
    json.dump(d, open(os.path.join(dd, "sample.json"), "w"), ensure_ascii=False, indent=2)
print(f"wrote {len(sub)} folders (err[{a}:{b}]+cor[{a}:{b}]) to {out}")
