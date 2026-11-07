import sys,time,pickle;sys.path.insert(0,'.')
from variants import *
which,B,setname=sys.argv[1],int(sys.argv[2]),sys.argv[3]
V=VARS1 if setname=='v1' else VARS2
models=fit_all(V)
d=prep(te) if which=='SEER' else prep(tc.dropna(subset=['M_stage']))
yref=Surv.from_arrays(tr_all.os_status.astype(bool),tr_all.survival_months) if which=='SEER' else Surv.from_arrays(d.os_status.astype(bool),d.survival_months)
P=predict(models,d); d=d.reset_index(drop=True)
names=list(V); n=len(d)
def metrics_for(idx):
    dd=d.iloc[idx]; y=Surv.from_arrays(dd.os_status.astype(bool),dd.survival_months); res={}
    for k in names:
        lp,S=P[k]; c=concordance_index_censored(dd.os_status.astype(bool),dd.survival_months,lp[idx])[0]
        auc=cumulative_dynamic_auc(yref,y,lp[idx],TIMES)[0]; _,br=brier_score(yref,y,S[idx],TIMES)
        res[k]=np.r_[c,auc,br]
    return res
full=metrics_for(np.arange(n)); pickle.dump(full,open(f'full_{which}_{setname}.pkl','wb'))
print(which,setname,'point estimates',{k:np.round(v,4).tolist() for k,v in full.items()},flush=True)
rng=np.random.default_rng(42); boots=[]; t0=time.time()
for b in range(B):
    idx=rng.integers(0,n,n)
    try: boots.append(metrics_for(idx))
    except Exception as e: continue
    if b%50==0: print(b,round(time.time()-t0),flush=True)
pickle.dump(boots,open(f'boot_{which}_{setname}.pkl','wb')); print('done',len(boots),flush=True)
