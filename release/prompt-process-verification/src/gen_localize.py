# -*- coding: utf-8 -*-
"""一个subset跑三prompt(v9t + v9num具体数字 + v9verify断言检验),存preds(new含base=v9t)。
用法: python opro_gen3.py CONFIG"""
import sys, re, json, random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
MODEL='models/DSR1-7B'; MAXLEN=16384
CONFIG=sys.argv[1]
random.seed(20240822); CONFIGS=['gsm8k','math','olympiadbench','omnimath']; K,START=60,50
tok=AutoTokenizer.from_pretrained(MODEL)
SYS="You are a meticulous math teacher checking a student's solution."

def head(problem,steps):
    numbered="\n".join(f"Step {i+1}: {s.strip()}" for i,s in enumerate(steps))
    return (f"{SYS}\n\nProblem:\n{problem}\n\nStudent's solution, numbered:\n{numbered}\n\n"
            f"Check the steps IN ORDER. For each step, verify that its result genuinely FOLLOWS from the "
            f"earlier steps: if it is a calculation, recompute it; if it is an algebraic manipulation or a "
            f"logical/deductive claim, check that the deduction is actually valid and justified. ")
TAIL=("Find the FIRST step that is DEFINITELY wrong. A step that is merely unsimplified, intermediate, a "
      "valid rearrangement, or a correct result written differently is NOT wrong. On the LAST line output "
      "exactly 'Answer: N' (the number of the first wrong step) or 'Answer: 0' if every step is correct.")
NUM=("Give extra scrutiny to any step that produces a SPECIFIC NUMBER that feeds into the final answer -- a "
     "total, a count, a solved value of a variable, an evaluated power/product, or a quantity then boxed or "
     "combined into the result. For each such number, recompute it explicitly from scratch and check TWO "
     "things: that the arithmetic is correct, AND that it is actually the quantity the problem asks for (not "
     "a different quantity confused for it). If a concrete number is miscomputed or is the wrong quantity, "
     "THAT step is the first wrong step. Apply this concrete-number recheck only to actual numeric results, "
     "not to conceptual claims. ")
VER=("Pay special attention to steps that ASSERT a conclusion without actually proving it -- for example "
     "'this construction works / satisfies the condition', 'this choice maximizes / is optimal', 'the only "
     "way is ...', 'this is the largest / the maximum is N', or a claimed formula, count, or complete list "
     "of cases. Do NOT accept such an assertion on its face: TEST it concretely before believing it. If a "
     "construction is claimed, check on a small explicit instance that it really satisfies EVERY condition "
     "in the problem. If a list of cases or a maximum is claimed complete, actively search for one more case "
     "or a larger value it missed. If a number is asserted, verify it. The moment such a concrete check "
     "fails, THAT step is the first wrong step. ")
def P_v9t(p,s): return tok.apply_chat_template([{"role":"user","content":head(p,s)+TAIL}],tokenize=False,add_generation_prompt=True)
def P_num(p,s): return tok.apply_chat_template([{"role":"user","content":head(p,s)+NUM+TAIL}],tokenize=False,add_generation_prompt=True)
def P_ver(p,s): return tok.apply_chat_template([{"role":"user","content":head(p,s)+VER+TAIL}],tokenize=False,add_generation_prompt=True)

def parse_robust(t,nsteps):
    body=re.split(r"</think>",t,flags=re.I)[-1]
    m=re.findall(r"first\s+wrong\s+step\s+is\s+step\s+(\d+)",body,re.I)
    m+=re.findall(r"(?:the\s+)?error\s+(?:first\s+)?(?:occurs?|is|appears)\s+(?:in|at)\s+step\s+(\d+)",body,re.I)
    m+=re.findall(r"step\s+(\d+)\s+is\s+(?:the\s+first\s+)?(?:step\s+that\s+is\s+)?(?:wrong|incorrect|erroneous|flawed)",body,re.I)
    for x in m:
        if 1<=int(x)<=nsteps: return int(x)-1
    ans=re.findall(r"answer\s*[:\-]?\s*\**\s*(\d+)",body,re.I)+re.findall(r"\boxed\{\s*(\d+)\s*\}",body)
    ans=[int(a) for a in ans if 0<=int(a)<=nsteps]
    if ans: return ans[-1]-1 if ans[-1]>0 else -1
    if re.search(r"all\s+(?:the\s+)?steps?\s+(?:are\s+)?correct|every\s+step\s+is\s+correct|no\s+(?:error|mistake)",body,re.I): return -1
    return -1

# 复现同一批120评测
for CF in CONFIGS:
    d=json.load(open(f"ProcessBench/{CF}.json"))
    ep=[x for x in d if x["label"]!=-1][START:]; cp=[x for x in d if x["label"]==-1][START:]
    er=random.sample(ep,min(K,len(ep))); co=random.sample(cp,min(K,len(cp)))
    if CF==CONFIG: cases=er+co
print(f"{CONFIG}: {len(cases)} cases",flush=True)
llm=LLM(model=MODEL,dtype="half",max_model_len=MAXLEN,gpu_memory_utilization=0.92,enforce_eager=True)
SPP=SamplingParams(temperature=0.0,max_tokens=5000,repetition_penalty=1.1)
allp=[P_v9t(x["problem"],x["steps"]) for x in cases]+[P_num(x["problem"],x["steps"]) for x in cases]+[P_ver(x["problem"],x["steps"]) for x in cases]
outs=llm.generate(allp,SPP); n=len(cases)
v9t=[parse_robust(outs[i].outputs[0].text,len(cases[i]["steps"])) for i in range(n)]
num=[parse_robust(outs[n+i].outputs[0].text,len(cases[i]["steps"])) for i in range(n)]
ver=[parse_robust(outs[2*n+i].outputs[0].text,len(cases[i]["steps"])) for i in range(n)]
def f1(P):
    eh=et=ch=ct=0
    for x,p in zip(cases,P):
        l=x["label"]
        if l==-1: ct+=1; ch+=(p==-1)
        else: et+=1; eh+=(p==l)
    ae=eh/max(et,1); ac=ch/max(ct,1); return round(2*ae*ac/max(ae+ac,1e-9)*100,1)
json.dump([{"id":cases[i]["id"],"label":cases[i]["label"],"pred":v9t[i]} for i in range(n)],open(f"{CONFIG}_v9t_preds.json","w"))
json.dump([{"id":cases[i]["id"],"label":cases[i]["label"],"base":v9t[i],"new":num[i]} for i in range(n)],open(f"{CONFIG}_v9num_preds.json","w"))
json.dump([{"id":cases[i]["id"],"label":cases[i]["label"],"base":v9t[i],"new":ver[i]} for i in range(n)],open(f"{CONFIG}_v9verify_preds.json","w"))
print(f"[{CONFIG}] v9t={f1(v9t)} v9num={f1(num)} v9verify={f1(ver)}",flush=True)
print("GEN3_DONE",flush=True)
