# -*- coding: utf-8 -*-
"""omnimath 稳健答案抽取(修 \boxed{N}占位符 / \boxed{题目答案}当步号 / prose-Answer不一致):
优先信 prose "first wrong step is Step N"，\boxed/Answer 仅在 0..nsteps 内才采信。v9t prompt。
用法: python opro_omnifix.py CONFIG(默认omnimath)
"""
import sys, re, json, random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = 'models/DSR1-7B'
TARGET = sys.argv[1] if len(sys.argv) > 1 else 'omnimath'
random.seed(20240822)
CONFIGS = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
GENPRM = {'gsm8k': 83.4, 'math': 80.0, 'olympiadbench': 72.3, 'omnimath': 71.5}
K, START, MAXLEN = 60, 50, 16384
tok = AutoTokenizer.from_pretrained(MODEL)
SYS = "You are a meticulous math teacher checking a student's solution."


def build_prompt(problem, steps):
    numbered = "\n".join(f"Step {i+1}: {s.strip()}" for i, s in enumerate(steps))
    user = (f"{SYS}\n\nProblem:\n{problem}\n\nStudent's solution, numbered:\n{numbered}\n\n"
            f"Check the steps IN ORDER. For each step, verify that its result genuinely FOLLOWS from the "
            f"earlier steps: if it is a calculation, recompute it; if it is an algebraic manipulation or a "
            f"logical/deductive claim, check that the deduction is actually valid and justified. Find the "
            f"FIRST step that is DEFINITELY wrong. A step that is merely unsimplified, intermediate, a valid "
            f"rearrangement, or a correct result written differently is NOT wrong. On the LAST line output "
            f"exactly 'Answer: N' (the number of the first wrong step) or 'Answer: 0' if every step is correct.")
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def parse_old(t):
    for pat in (r"answer\s*[:\-]?\s*\**\s*(\d+)", r"\\boxed\{\s*(\d+)\s*\}"):
        m = re.findall(pat, t, re.I)
        if m:
            num = int(m[-1]); return num - 1 if num > 0 else -1
    if re.search(r"all\s+(?:the\s+)?steps?\s+(?:are\s+)?correct|every\s+step\s+is\s+correct", t, re.I):
        return -1
    return -1


def parse_robust(t, nsteps):
    body = re.split(r"</think>", t, flags=re.I)[-1]
    # 1) prose 明确点名首错步(最可信)
    m = re.findall(r"first\s+wrong\s+step\s+is\s+step\s+(\d+)", body, re.I)
    m += re.findall(r"(?:the\s+)?error\s+(?:first\s+)?(?:occurs?|is|appears)\s+(?:in|at)\s+step\s+(\d+)", body, re.I)
    m += re.findall(r"step\s+(\d+)\s+is\s+(?:the\s+first\s+)?(?:step\s+that\s+is\s+)?(?:wrong|incorrect|erroneous|flawed)", body, re.I)
    for x in m:
        if 1 <= int(x) <= nsteps:
            return int(x) - 1
    # 2) Answer/boxed，但只在 0..nsteps 内(拒绝题目答案/占位符)
    ans = re.findall(r"answer\s*[:\-]?\s*\**\s*(\d+)", body, re.I) + re.findall(r"\\boxed\{\s*(\d+)\s*\}", body)
    ans = [int(a) for a in ans if 0 <= int(a) <= nsteps]
    if ans:
        return ans[-1] - 1 if ans[-1] > 0 else -1
    # 3) 明确全对
    if re.search(r"all\s+(?:the\s+)?steps?\s+(?:are\s+)?correct|every\s+step\s+is\s+correct|no\s+(?:error|mistake)", body, re.I):
        return -1
    return -1


llm = LLM(model=MODEL, dtype="half", max_model_len=MAXLEN, gpu_memory_utilization=0.92, enforce_eager=True)
SP = SamplingParams(temperature=0.0, max_tokens=5000, repetition_penalty=1.1)

for CONFIG in CONFIGS:
    d = json.load(open(f"ProcessBench/{CONFIG}.json"))
    epool = [x for x in d if x["label"] != -1][START:]; cpool = [x for x in d if x["label"] == -1][START:]
    err = random.sample(epool, min(K, len(epool))); cor = random.sample(cpool, min(K, len(cpool)))
    if CONFIG == TARGET:
        cases = err + cor


def f1_of(preds):
    eh = et = ch = ct = 0
    for x, p in zip(cases, preds):
        e = x["label"]
        if e == -1:
            ct += 1; ch += (p == -1)
        else:
            et += 1; eh += (p == e)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return round(2 * ae * ac / max(ae + ac, 1e-9) * 100, 1), eh, et, ch, ct


outs = llm.generate([build_prompt(x["problem"], x["steps"]) for x in cases], SP)
texts = [o.outputs[0].text for o in outs]
_CONF = re.compile(r"(?:mistake|error|flaw)\s+(?:is\s+)?in\s+step\s+(\d+)|step\s+(\d+)\s+(?:is|has|contains|makes)\s+(?:an?\s+)?(?:definit\w+\s+)?(?:wrong|incorrect|erroneous|error|mistake|invalid)", re.I)


def parse_escalate(t, nsteps):
    p = parse_robust(t, nsteps)
    if p != -1:
        return p
    think = re.split(r"</think>", t, flags=re.I)[0]
    hits = []
    for m in _CONF.finditer(think):
        s = m.group(1) or m.group(2)
        if s and 1 <= int(s) <= nsteps:
            hits.append(int(s) - 1)
    return hits[-1] if hits else -1


old = [parse_old(t) for t in texts]
new = [parse_robust(t, len(x["steps"])) for x, t in zip(cases, texts)]
esc = [parse_escalate(t, len(x["steps"])) for x, t in zip(cases, texts)]
fo = f1_of(old); fn = f1_of(new); fe = f1_of(esc)
rescued = sum(1 for x, po, pn in zip(cases, old, new) if po != x["label"] and pn == x["label"])
broke = sum(1 for x, po, pn in zip(cases, old, new) if po == x["label"] and pn != x["label"])
print(f"[{TARGET}] 旧解析 F1={fo[0]} (错{fo[1]}/{fo[2]} 对{fo[3]}/{fo[4]})", flush=True)
print(f"[{TARGET}] 稳健解析 F1={fn[0]} (错{fn[1]}/{fn[2]} 对{fn[3]}/{fn[4]})  救回{rescued} 弄坏{broke}  (GenPRM={GENPRM[TARGET]})", flush=True)
er = sum(1 for x, pn, pe in zip(cases, new, esc) if pn != x["label"] and pe == x["label"])
eb = sum(1 for x, pn, pe in zip(cases, new, esc) if pn == x["label"] and pe != x["label"])
print(f"[{TARGET}] 稳健+回挖 F1={fe[0]} (错{fe[1]}/{fe[2]} 对{fe[3]}/{fe[4]})  在稳健上救回{er} 弄坏{eb}", flush=True)
print("FIX_DONE", flush=True)
