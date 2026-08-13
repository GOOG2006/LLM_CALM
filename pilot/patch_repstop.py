"""给 genprm_inference.py 的 analyze 阶段挂一个'复读早停' logits processor。
只在检测到近似周期性复读时强制 EOS 收尾;正常生成不触发 -> 其他子集输出字节级不变。
容错周期检测:对周期 p,若最近 3 个长度 p 的块两两只差 <=tol 个 token(容忍递增的段号/数字),判为循环。
生成 genprm_inference.py(原地改),备份 .bak_repstop。"""
F = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py"
s = open(F, encoding="utf-8").read()

CLS = '''

class _RepStop:
    """Force EOS when the generated tail shows near-periodic repetition (degenerate loop).
    Fires ONLY on real repetition -> non-repetitive generations are byte-identical."""
    def __init__(self, eos_id, min_len=120, max_period=48, tol=2, reps=3):
        self.eos = eos_id
        self.min_len = min_len
        self.max_period = max_period
        self.tol = tol
        self.reps = reps
    def __call__(self, past, logits):
        n = len(past)
        if self.eos is None or n < self.min_len:
            return logits
        for p in range(8, self.max_period + 1):
            if n < self.reps * p:
                break
            b0 = past[-p:]
            ok = True
            for r in range(2, self.reps + 1):
                br = past[-p * r:-p * (r - 1)]
                if sum(1 for x, y in zip(b0, br) if x != y) > self.tol:
                    ok = False
                    break
            if ok:
                logits[:] = float('-inf')
                logits[self.eos] = 0.0
                return logits
        return logits
'''

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, f"expected {n} of <<{old[:50]}>> got {s.count(old)}"
    s = s.replace(old, new)

# 1) 在 version 常量后插入类定义
rep("version = 'v1.0'\n", "version = 'v1.0'\n" + CLS)

# 2) __init__ 里 tokenizer 之后建实例
rep("        self.tokenizer = AutoTokenizer.from_pretrained(model_path)\n",
    "        self.tokenizer = AutoTokenizer.from_pretrained(model_path)\n"
    "        self._repstop = _RepStop(self.tokenizer.eos_token_id)\n")

# 3) 仅给 analyze 阶段的 SamplingParams 挂上(唯一的 'REPETITION_PENALTY' 后不带逗号者)
rep("                logprobs=20,  # Number of log probabilities to return\n"
    "                repetition_penalty=REPETITION_PENALTY\n"
    "            )",
    "                logprobs=20,  # Number of log probabilities to return\n"
    "                repetition_penalty=REPETITION_PENALTY,\n"
    "                logits_processors=[self._repstop]\n"
    "            )")

open(F, "w", encoding="utf-8").write(s)
print("patched:", all(x in s for x in ["class _RepStop", "self._repstop = _RepStop", "logits_processors=[self._repstop]"]))
