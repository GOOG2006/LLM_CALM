"""复读早停 v2:用'逐字 n-gram 复读率'判据(区分'合法结构化验算'与'退化复读')。
- 只在最近 window 个 token 的 4-gram 复读率 > thr 时强制 EOS。
- 合法逐项验算(每行数字不同)-> 4gram 复读率≈0 -> 不触发;退化复读(整句逐字重复)-> 高 -> 触发。
- 触发时把 (复读率 + 解码片段) 写进 repstop_fire.log 便于验证。
基于 .bak_repstop(干净:crash-safe + temp0.6 + rep1.0,无 repstop)生成。"""
BAK = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py.bak_repstop"
DST = "/root/autodl-tmp/genprm_work/GenPRM/src/prm_evaluation/genprm_inference.py"
s = open(BAK, encoding="utf-8").read()

CLS = '''

class _RepStop:
    """Force EOS when the recent window shows high VERBATIM n-gram repetition (degenerate loop).
    Legit structured verification (each line has different numbers) has ~0 n-gram repeat -> never fires.
    True loops (a phrase repeated verbatim) have high n-gram repeat -> fires."""
    def __init__(self, eos_id, tokenizer=None, min_len=200, window=120, n=4, thr=0.5):
        self.eos = eos_id
        self.tok = tokenizer
        self.min_len = min_len
        self.window = window
        self.n = n
        self.thr = thr
    def __call__(self, past, logits):
        L = len(past)
        if self.eos is None or L < self.min_len:
            return logits
        w = past[-self.window:]
        grams = [tuple(w[i:i + self.n]) for i in range(len(w) - self.n + 1)]
        if not grams:
            return logits
        rep = 1.0 - len(set(grams)) / len(grams)
        if rep > self.thr:
            try:
                snip = self.tok.decode(past[-80:]) if self.tok is not None else ''
                open('/root/autodl-tmp/genprm_work/repstop_fire.log', 'a').write(('%.2f\\t%s' % (rep, snip[-160:].replace(chr(10), ' '))) + chr(10))
            except Exception:
                pass
            logits[:] = float('-inf')
            logits[self.eos] = 0.0
        return logits
'''

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, f"expected {n} of <<{old[:50]}>> got {s.count(old)}"
    s = s.replace(old, new)

rep("version = 'v1.0'\n", "version = 'v1.0'\n" + CLS)
rep("        self.tokenizer = AutoTokenizer.from_pretrained(model_path)\n",
    "        self.tokenizer = AutoTokenizer.from_pretrained(model_path)\n"
    "        self._repstop = _RepStop(self.tokenizer.eos_token_id, self.tokenizer)\n")
rep("                logprobs=20,  # Number of log probabilities to return\n"
    "                repetition_penalty=REPETITION_PENALTY\n"
    "            )",
    "                logprobs=20,  # Number of log probabilities to return\n"
    "                repetition_penalty=REPETITION_PENALTY,\n"
    "                logits_processors=[self._repstop]\n"
    "            )")

open(DST, "w", encoding="utf-8").write(s)
print("patched v2:", all(x in s for x in ["high VERBATIM n-gram", "self._repstop = _RepStop(self.tokenizer.eos_token_id, self.tokenizer)", "logits_processors=[self._repstop]"]))
