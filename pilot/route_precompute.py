"""用裸 DSR1(无代码偏见)预计算 per-step 路由:每步判"该不该用代码验证",读 Yes/No logit。
保存 {case_id: [use_code bool per step]} 供 adaptive 用,并报每子集代码率。
用法: python route_precompute.py <model> <config> <n> <out_json>"""
import os, json, sys, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model, config, N, outj = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
data = json.load(open(f"/root/autodl-tmp/genprm_work/ProcessBench/{config}.json"))
err = [d for d in data if d["label"] != -1][: N // 2]
cor = [d for d in data if d["label"] == -1][: N - N // 2]
cases = err + cor

PROMPT = ("You are deciding how to verify ONE step of a math solution. Running Python code helps ONLY "
          "for concrete numeric or algebraic CALCULATIONS (arithmetic, evaluating an expression, solving a "
          "specific equation). It does NOT help for proofs, logical arguments, definitions, case analysis, or "
          "merely setting up equations.\n\nStep: {step}\n\nShould we run Python code to check THIS step? "
          "Answer with one word: Yes or No.\nAnswer:")

tok = AutoTokenizer.from_pretrained(model)
def tid(w):
    for v in (w, " " + w):
        e = tok(v, add_special_tokens=False).input_ids
        if e: return e[0]
yes_ids = {tid("Yes"), tid("yes"), tid("Correct")}; no_ids = {tid("No"), tid("no")}
yes_ids.discard(None); no_ids.discard(None)

llm = LLM(model=model, dtype="half", max_model_len=4096, gpu_memory_utilization=0.6,
          enforce_eager=True, swap_space=0)
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

prompts, meta = [], []
for d in cases:
    for si, st in enumerate(d["steps"]):
        prompts.append(PROMPT.format(step=(st or "")[:1500])); meta.append((d["id"], si))
outs = llm.generate(prompts, sp)
routing = {}
for (cid, si), o in zip(meta, outs):
    lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
    py = sum(math.exp(v.logprob) for k, v in lp.items() if k in yes_ids)
    pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in no_ids)
    routing.setdefault(cid, []).append(bool(py >= pn))
json.dump(routing, open(outj, "w"))
flat = [u for v in routing.values() for u in v]
print(f"{config}: {len(cases)}案 {len(flat)}步  代码率={100*sum(flat)/max(len(flat),1):.0f}%  -> {outj}")
