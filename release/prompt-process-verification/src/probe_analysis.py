# -*- coding: utf-8 -*-
"""omnimath 域内隐探针:base模型层-l 步末token隐状态 → 步正确性。训练=非评测案例(严格不相交),
层/阈值在训练内部val选,报评测120例的定位F1 + 与v9t(67.8)融合。"""
import json, random, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL='models/DSR1-7B'; MAXLEN=4096
CAND=[-5,-7,-9,-11,-13]
random.seed(20240822); np.random.seed(0); torch.manual_seed(0)

data=json.load(open("ProcessBench/omnimath.json"))
v9t=json.load(open("omni_v9t_preds.json"))
eval_ids=[c["id"] for c in v9t]; eval_set=set(eval_ids)
byid={x["id"]:x for x in data}
eval_cases=[byid[i] for i in eval_ids]                      # 120, 与v9t同序
v9t_base=[c["pred"] for c in v9t]; v9t_lab=[c["label"] for c in v9t]
trpool=[x for x in data if x["id"] not in eval_set]
tr_err=[x for x in trpool if x["label"]!=-1]; tr_cor=[x for x in trpool if x["label"]==-1]
random.shuffle(tr_err); random.shuffle(tr_cor)
train_cases=tr_err[:500]+tr_cor[:181]
random.shuffle(train_cases)
val_cases=train_cases[:120]; fit_cases=train_cases[120:]   # 严格切分:val不进训练
print(f"fit={len(fit_cases)} val={len(val_cases)} eval={len(eval_cases)}",flush=True)

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16).cuda().eval()

def feats(problem, steps):
    steps=[s for s in steps if s is not None]
    if steps and steps[-1]=='': steps=steps[:-1]
    fid=list(tok(problem+"\n",add_special_tokens=True).input_ids); bnd=[]
    for s in steps:
        st=len(fid); fid+=tok(s+"\n",add_special_tokens=False).input_ids; bnd.append((st,len(fid)))
    if len(fid)>MAXLEN:
        fid=fid[:MAXLEN]; bnd=[(a,min(b,MAXLEN)) for a,b in bnd if a<MAXLEN]
    if len(bnd)<1: return None
    ids=torch.tensor([fid],device='cuda')
    with torch.no_grad():
        hs=model(ids,output_hidden_states=True).hidden_states  # tuple len 29
        out={}
        for L in CAND:
            h=hs[L][0]
            out[L]=torch.stack([h[b-1] for (a,b) in bnd]).float().cpu().numpy()
    return out  # dict L->[nsteps,H]

def steplabels(x, nsteps):
    e=x["label"]
    if e==-1: return list(range(nsteps)),[0]*nsteps
    e=min(e,nsteps-1)
    return list(range(e+1)),[0]*e+[1]

# ---- 抽训练特征(仅fit)----
TRX={L:[] for L in CAND}; TRY=[]
for k,x in enumerate(fit_cases):
    F=feats(x["problem"],x["steps"])
    if F is None: continue
    ns=F[CAND[0]].shape[0]; idxs,labs=steplabels(x,ns)
    idxs=[i for i in idxs if i<ns]; labs=labs[:len(idxs)]
    for L in CAND: TRX[L].append(F[L][idxs])
    TRY.extend(labs)
    if (k+1)%100==0: print(f"  train feat {k+1}/{len(fit_cases)}",flush=True)
TRY=np.array(TRY,dtype=np.float32)
for L in CAND: TRX[L]=np.concatenate(TRX[L],0).astype(np.float32)
print(f"train steps={len(TRY)} wrong={int(TRY.sum())} correct={int((TRY==0).sum())}",flush=True)

# ---- 抽评测特征 ----
EV=[]
for x in eval_cases:
    EV.append(feats(x["problem"],x["steps"]))
print("eval feats done",flush=True)

# ---- 每层训练logreg,训练内部val选层+阈值 ----
def train_logreg(X,y,mu,sd):
    Xz=(X-mu)/sd
    Xt=torch.tensor(Xz,device='cuda'); yt=torch.tensor(y,device='cuda')
    w=torch.zeros(X.shape[1],device='cuda',requires_grad=True); b=torch.zeros(1,device='cuda',requires_grad=True)
    pw=torch.tensor((1-y.mean())/max(y.mean(),1e-3),device='cuda')
    opt=torch.optim.Adam([w,b],lr=0.05); lf=torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(400):
        opt.zero_grad(); loss=lf(Xt@w+b,yt)+1e-2*(w*w).sum(); loss.backward(); opt.step()
    return w.detach(),b.detach()

def loc_f1(cases_lab, preds):
    eh=et=ch=ct=0
    for lab,p in zip(cases_lab,preds):
        if lab==-1: ct+=1; ch+=(p==-1)
        else: et+=1; eh+=(p==lab)
    ae=eh/max(et,1); ac=ch/max(ct,1); return 2*ae*ac/max(ae+ac,1e-9)*100

# val_cases 已在开头切分(不在fit内),现场抽其定位特征
def predict_layer(Fdict,L,w,b,mu,sd,tau,zmode):
    if Fdict is None: return -1
    Fz=(Fdict[L]-mu)/sd
    sc=torch.sigmoid(torch.tensor(Fz,device='cuda',dtype=torch.float32)@w+b).cpu().numpy()
    if zmode:
        z=(sc-sc.mean())/(sc.std()+1e-9)
        return next((i for i,v in enumerate(z) if v>tau),-1)
    return next((i for i,v in enumerate(sc) if v>tau),-1)

best=(-1,None)  # (f1, cfg)
val_feat=[feats(x["problem"],x["steps"]) for x in val_cases]
val_lab=[x["label"] for x in val_cases]
for L in CAND:
    mu=TRX[L].mean(0); sd=TRX[L].std(0)+1e-6
    w,b=train_logreg(TRX[L],TRY,mu,sd)
    for zmode in [False,True]:
        taus=[i/20 for i in range(0,20)] if not zmode else [i/10 for i in range(-5,31)]
        for tau in taus:
            preds=[predict_layer(F,L,w,b,mu,sd,tau,zmode) for F in val_feat]
            f=loc_f1(val_lab,preds)
            if f>best[0]: best=(f,(L,zmode,tau,mu,sd,w,b))
    print(f"  layer{L} done, best-so-far valF1={best[0]:.1f} cfg L={best[1][0]} z={best[1][1]} tau={best[1][2]}",flush=True)

L,zmode,tau,mu,sd,w,b=best[1]
print(f"[SELECTED] layer={L} zmode={zmode} tau={tau} (train-val F1={best[0]:.1f})",flush=True)

# ---- 评测 ----
probe_pred=[predict_layer(F,L,w,b,mu,sd,tau,zmode) for F in EV]
# 对齐label边界
probe_pred=[p if (p==-1 or (EV[i] is not None and p<EV[i][L].shape[0])) else -1 for i,p in enumerate(probe_pred)]
f_v9t=loc_f1(v9t_lab,v9t_base)
f_probe=loc_f1(v9t_lab,probe_pred)
# 融合A: v9t判-1时用探针补; 否则信v9t
fuseA=[v9t_base[i] if v9t_base[i]!=-1 else probe_pred[i] for i in range(len(EV))]
f_fuseA=loc_f1(v9t_lab,fuseA)
# oracle上界(评测上扫tau)
def oracle():
    bestf=0
    for zmode2 in [False,True]:
        taus=[i/20 for i in range(0,20)] if not zmode2 else [i/10 for i in range(-5,31)]
        for t in taus:
            pp=[predict_layer(F,L,w,b,mu,sd,t,zmode2) for F in EV]
            pp=[p if (p==-1 or (EV[i] is not None and p<EV[i][L].shape[0])) else -1 for i,p in enumerate(pp)]
            fa=[v9t_base[i] if v9t_base[i]!=-1 else pp[i] for i in range(len(EV))]
            bestf=max(bestf,loc_f1(v9t_lab,fa))
    return bestf
print(f"v9t(baseline)   F1={f_v9t:.1f}",flush=True)
print(f"probe-alone     F1={f_probe:.1f}",flush=True)
print(f"fuseA(v9t+probe补-1) F1={f_fuseA:.1f}",flush=True)
print(f"fuseA oracle上界 F1={oracle():.1f}",flush=True)
# 探针在v9t漏判(8个-1错题)上的定位
miss=[i for i in range(len(EV)) if v9t_lab[i]!=-1 and v9t_base[i]==-1]
hit=sum(1 for i in miss if probe_pred[i]==v9t_lab[i])
print(f"v9t漏判{len(miss)}个中探针定位对={hit}",flush=True)
json.dump({"probe_pred":probe_pred,"labels":v9t_lab,"base":v9t_base,"sel":[L,zmode,tau]},
          open("omni_probe_indom.json","w"))
print("OMNIPROBE_DONE",flush=True)
