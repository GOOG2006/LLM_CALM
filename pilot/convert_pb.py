"""ProcessBench json -> 官方 prm_evaluate 需要的 <folder>/sample.json 结构。
用法: python convert_pb.py gsm8k 100 /root/autodl-tmp/genprm_work/pb_gsm8k_in"""
import json, os, sys
config, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = json.load(open(f"/root/autodl-tmp/genprm_work/ProcessBench/{config}.json"))
err = [d for d in data if d["label"] != -1][: n // 2]
cor = [d for d in data if d["label"] == -1][: n - n // 2]
sub = err + cor
os.makedirs(out, exist_ok=True)
for i, d in enumerate(sub):
    dd = os.path.join(out, f"{config}_{i:03d}")
    os.makedirs(dd, exist_ok=True)
    json.dump(d, open(os.path.join(dd, "sample.json"), "w"), ensure_ascii=False, indent=2)
print(f"wrote {len(sub)} folders ({len(err)} err / {len(cor)} cor) to {out}")
