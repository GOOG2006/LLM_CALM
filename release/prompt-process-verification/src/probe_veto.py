# -*- coding: utf-8 -*-
"""某subset的 clause∩probe-veto。用法: python opro_probeveto_sub.py CONFIG"""
import sys, json, random, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL='models/DSR1-7B'; MAXLEN=4096; CAND=[-7,-9]; CONFIG=sys.argv[1]
random.seed(20240822); np.random.seed(0); torch.manual_seed(0)
data=json.load(open(f"ProcessBench/{CONFIG}.json"))
v9t=json.load(open(f"{CONFIG}_v9t_preds.json"))
eval_ids=[c["id"] for c in v9t]; eval_set=set(eval_ids); byid={x["id"]:x for x in data}
eval_cases=[byid[i] for i in eval_ids]; base=[c["pred"] for c in v9t]; lab=[c["label"] for c in v9t]
num={x["id"]:x["new"] for x in json.load(open(f"{CONFIG}_v9num_preds.json"))}
ver={x["id"]:x["new"] for x in json.load(open(f"{CONFIG}_v9verify_preds.json"))}
clause=[]
for c in v9t:
    b=c["pred"]
    if b!=-1: clause.append(b)
    else:
        nn=num.get(c["id"],-1); vv=ver.get(c["id"],-1); clause.append(nn if nn!=-1 else vv)
trpool=[x for x in data if x["id"] not in eval_set]
tr_err=[x for x in trpool if x["label"]!=-1]; tr_cor=[x for x in trpool if x["label"]==-1]
random.shuffle(tr_err); random.shuffle(tr_cor)
fit_cases=tr_err[:500]+tr_cor[:min(200,len(tr_cor))]; random.shuffle(fit_cases)
print(f"{CONFIG}: fit={len(fit_cases)} eval={len(eval_cases)} tr_cor={len(tr_cor)}",flush=True)
tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16).cuda().eval()
def feats(problem,steps):
    steps=[s for s in steps if s is not None]
    if steps and steps[-1]=='': steps=steps[:-1]
    fid=list(tok(problem+"\n",add_special_tokens=True).input_ids); bnd=[]
    for s in steps:
        st=len(fid); fid+=tok(s+"\n",add_special_tokens=False).input_ids; bnd.append((st,len(fid)))
    if len(fid)>MAXLEN: fid=fid[:MAXLEN]; bnd=[(a,min(b,MAXLEN)) for a,b in bnd if a<MAXLEN]
    if len(bnd)<1: return None
    ids=torch.tensor([fid],device='cuda')
    with torch.no_grad():
        hs=model(ids,output_hidden_states=True).hidden_states
        return {L:torch.stack([hs[L][0][b-1] for (a,b) in bnd]).float().cpu().numpy() for L in CAND}
def steplabels(x,ns):
    e=x["label"]
    if e==-1: return list(range(ns)),[0]*ns
    e=min(e,ns-1); return list(range(e+1)),[0]*e+[1]
TRX={L:[] for L in CAND}; TRY=[]
for k,x in enumerate(fit_cases):
    F=feats(x["problem"],x["steps"])
    if F is None: continue
    ns=F[CAND[0]].shape[0]; idxs,labs=steplabels(x,ns); idxs=[i for i in idxs if i<ns]; labs=labs[:len(idxs)]
    for L in CAND: TRX[L].append(F[L][idxs])
    TRY.extend(labs)
    if (k+1)%200==0: print(f"  feat {k+1}",flush=True)
TRY=np.array(TRY,np.float32)
for L in CAND: TRX[L]=np.concatenate(TRX[L],0).astype(np.float32)
EV=[feats(x["problem"],x["steps"]) for x in eval_cases]
print("feats done",flush=True)
def train_logreg(X,y,mu,sd):
    Xz=(X-mu)/sd; Xt=torch.tensor(Xz,device='cuda'); yt=torch.tensor(y,device='cuda')
    w=torch.zeros(X.shape[1],device='cuda',requires_grad=True); b=torch.zeros(1,device='cuda',requires_grad=True)
    pw=torch.tensor((1-y.mean())/max(y.mean(),1e-3),device='cuda')
    opt=torch.optim.Adam([w,b],lr=0.05); lf=torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(400): opt.zero_grad(); loss=lf(Xt@w+b,yt)+1e-2*(w*w).sum(); loss.backward(); opt.step()
    return w.detach(),b.detach()
def sscore(Fdict,L,w,b,mu,sd):
    if Fdict is None: return None
    Fz=(Fdict[L]-mu)/sd
    return torch.sigmoid(torch.tensor(Fz,device='cuda',dtype=torch.float32)@w+b).cpu().numpy()
def f1(P):
    eh=et=ch=ct=0
    for l,p in zip(lab,P):
        if l==-1: ct+=1; ch+=(p==-1)
        else: et+=1; eh+=(p==l)
    ae=eh/max(et,1); ac=ch/max(ct,1); return round(2*ae*ac/max(ae+ac,1e-9)*100,1),eh,et,ch,ct
print("参考: v9t=%s | 条款并集=%s"%(f1(base)[0],f1(clause)[0]),flush=True)
for L in CAND:
    mu=TRX[L].mean(0); sd=TRX[L].std(0)+1e-6; w,b=train_logreg(TRX[L],TRY,mu,sd)
    evs=[sscore(F,L,w,b,mu,sd) for F in EV]
    for topj in [1,2,3]:
        pred=[]
        for i in range(len(base)):
            if base[i]!=-1: pred.append(base[i]); continue
            k=clause[i]
            if k==-1: pred.append(-1); continue
            s=evs[i]
            if s is None or k>=len(s): pred.append(-1); continue
            pred.append(k if k in list(np.argsort(-s)[:topj]) else -1)
        ff=f1(pred)
        res=sum(1 for i in range(len(base)) if base[i]!=lab[i] and pred[i]==lab[i])
        brk=sum(1 for i in range(len(base)) if base[i]==lab[i] and pred[i]!=lab[i])
        print("[%s L=%d top%d] F1=%s (err%d/%d cor%d/%d) +%d/-%d"%(CONFIG,L,topj,ff[0],ff[1],ff[2],ff[3],ff[4],res,brk),flush=True)
print("VETOSUB_DONE",flush=True)
