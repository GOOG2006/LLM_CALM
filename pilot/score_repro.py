"""给官方 prm_evaluate 输出的文件夹算 ProcessBench F1(可对部分完成结果算,中间报告用)。
用法: python score_repro.py <out_dir>"""
import sys, json, glob, os

outdir = sys.argv[1]
files = []
for d in sys.argv[1:]:  # 支持多目录合并
    files += glob.glob(os.path.join(d, "*", "result_*.json"))
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
        try:
            s = float(s)
        except Exception:
            s = 0.5
        if s < thr:
            return i
    return -1

err = [r for r in rows if int(r["label"]) != -1]
cor = [r for r in rows if int(r["label"]) == -1]
rec = sum(pred(r["value"]) == int(r["label"]) for r in err) / max(len(err), 1)
ca = sum(pred(r["value"]) == -1 for r in cor) / max(len(cor), 1)
f1 = 2 * rec * ca / max(rec + ca, 1e-9)
print(f"完成 {len(rows)} 案 (err {len(err)} / cor {len(cor)})")
print(f"err-acc(定位对首错)={rec:.3f}  cor-acc(全对判全对)={ca:.3f}  F1={f1*100:.1f}")
print(f"对照: 论文 GenPRM-1.5B MATH Pass@1=66.6 | 我们自制脚本(整解答格式)=45.4")
