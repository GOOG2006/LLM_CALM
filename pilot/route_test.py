"""离线测路由规则的分化力(纯文本,无GPU):对各子集 step 文本算代码率。
目标:计算型子集(gsm8k/omnimath/math)代码率高,证明型(olympiad)代码率低。"""
import re, json

def load_steps(cfg, n=60):
    d = json.load(open(f"/root/autodl-tmp/genprm_work/ProcessBench/{cfg}.json"))
    err = [x for x in d if x["label"] != -1][:n//2]; cor = [x for x in d if x["label"] == -1][:n-n//2]
    return [s for x in err+cor for s in x["steps"]]

NUM = r'\d[\d,]*\.?\d*'
routers = {
    # v0 旧:任意 数字-运算符-数字 或 =数字(太粗)
    "v0_any": lambda t: bool(re.search(NUM+r'\s*[\+\-\*/×÷]\s*'+NUM, t) or re.search(r'=\s*'+NUM, t)),
    # v1 完整算式:数字 运算 数字 ... = 数字(有可核验的算术结果)
    "v1_complete": lambda t: bool(re.search(NUM+r'\s*[\+\-\*/×·÷]\s*'+NUM+r'.{0,25}?=\s*'+NUM, t)),
    # v2 完整算式 且 不含证明/变量线索
    "v2_complete_noproof": lambda t: bool(re.search(NUM+r'\s*[\+\-\*/×·÷]\s*'+NUM+r'.{0,25}?=\s*'+NUM, t))
        and not re.search(r'\b(prove|proof|since|suppose|assume|consider|let\b|for all|there exist|WLOG|contradiction|modul|inequalit|divisib)', t, re.I),
    # v3 数字密度:算式结果占比高(≥2个"=数字")
    "v3_multi_eqnum": lambda t: len(re.findall(r'=\s*'+NUM, t)) >= 2,
}
PROOF = re.compile(r'\b(prove|proof|since|suppose|assume|consider|for all|there exist|WLOG|contradiction|'
                   r'modul|inequalit|divisib|theorem|lemma|denote|arbitrary|property|properties|'
                   r'function|integer|odd|even|prime|induction|general)', re.I)
routers.update({
    # 纯排除:没有证明/概念语言 → 用代码
    "v4_noproof": lambda t: not PROOF.search(t or ''),
    # 有"=数字" 且 无证明语言
    "v5_eqnum_noproof": lambda t: bool(re.search(r'=\s*'+NUM, t)) and not PROOF.search(t or ''),
    # 有 数字-运算符-数字 且 无证明语言
    "v6_arith_noproof": lambda t: bool(re.search(NUM+r'\s*[\+\-\*/×÷·]\s*'+NUM, t)) and not PROOF.search(t or ''),
    # 完整算式 或 (有=数字 且 无证明)
    "v7_complete_or_eqnp": lambda t: bool(re.search(NUM+r'\s*[\+\-\*/×·÷]\s*'+NUM+r'.{0,25}?=\s*'+NUM, t))
        or (bool(re.search(r'=\s*'+NUM, t)) and not PROOF.search(t or '')),
})

cfgs = ["gsm8k", "math", "omnimath", "olympiadbench"]
steps = {c: load_steps(c) for c in cfgs}
print(f"{'router':<22}" + "".join(f"{c:>14}" for c in cfgs) + "   gap(计算-olymp)")
for name, fn in routers.items():
    rates = {c: 100*sum(fn(s) for s in steps[c])/max(len(steps[c]),1) for c in cfgs}
    comp = (rates['gsm8k']+rates['math']+rates['omnimath'])/3
    print(f"{name:<22}" + "".join(f"{rates[c]:>13.0f}%" for c in cfgs) + f"   {comp-rates['olympiadbench']:+.0f}")
