# -*- coding: utf-8 -*-
"""CPU-only replay: validate the SAFE-COMBINATION machinery (gate + union + probe-veto)
on cached OmniMATH predictions, with NO GPU and NO analyst LLM.

It replays the two clauses the loop discovered by hand (concrete-number, assertion/
enumeration) through combiner.py and reproduces:  base 67.8 -> union 69.6 -> +veto 71.5.
The probe-veto step runs only if sample_data/omnimath_probe_scores_L-7.json is present
(that file is produced by a GPU run of ../src/probe_veto.py; absent here since the box is off).
"""
import os, json
import combiner as C

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "sample_data")

num = C.load(os.path.join(D, "omnimath_clause_concrete_number.json"))
ver = C.load(os.path.join(D, "omnimath_clause_assertion.json"))
assert [x["id"] for x in num] == [x["id"] for x in ver], "prediction files must align by id"

labels = [x["label"] for x in num]
base = [x["base"] for x in num]           # frozen v9t localizer
clause_num = [x["new"] for x in num]
clause_ver = [x["new"] for x in ver]

print("=== CPU replay: safe-combination machinery on OmniMATH (N=120) ===")
print(f"localizer base (v9t)         F1={C.f1(labels, base)[0]}")

# each clause GATED (fill misses only)
g_num = C.gated(base, clause_num)
g_ver = C.gated(base, clause_ver)
print(f"+concrete-number  (gated)    F1={C.f1(labels, g_num)[0]}  "
      f"(+{len(C.budget(base,labels,g_num)[0])}/-{len(C.budget(base,labels,g_num)[1])})")
print(f"+assertion/enum   (gated)    F1={C.f1(labels, g_ver)[0]}  "
      f"(+{len(C.budget(base,labels,g_ver)[0])}/-{len(C.budget(base,labels,g_ver)[1])})")

# prediction-level UNION (num-first: larger net first)
order = sorted([0, 1], key=lambda j: -len(C.budget(base, labels,
              C.gated(base, [clause_num, clause_ver][j]))[0]))
union = C.union(base, [clause_num, clause_ver], order=order)
ru, bu = C.budget(base, labels, union)
print(f"union of both clauses        F1={C.f1(labels, union)[0]}  (+{len(ru)}/-{len(bu)})  "
      f"rescued={[num[i]['id'].split('-')[1] for i in ru]}")

# optional PROBE-VETO (needs cached probe scores from a GPU run)
pv_path = os.path.join(D, "omnimath_probe_scores_L-7.json")
if os.path.exists(pv_path):
    probe = C.load(pv_path)
    for topk in (1, 2, 3):
        pv = C.probe_veto(union, base, probe, topk=topk)
        rp, bp = C.budget(base, labels, pv)
        print(f"union & probe-veto top{topk}   F1={C.f1(labels, pv)[0]}  (+{len(rp)}/-{len(bp)})")
else:
    print("union & probe-veto           [skipped: no cached probe scores; run ../src/probe_veto.py "
          "on the GPU to produce sample_data/omnimath_probe_scores_L-7.json -> expected 71.5]")
