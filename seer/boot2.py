import sys,time,pickle;sys.path.insert(0,'.')
from metrics import *
which,B,seed=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
P=pickle.load(open('preds_all.pkl','rb'))[which]
tr_=prep(tr); d=(prep(te) if which=='SEER' else prep(tc.dropna(subset=['M_stage']))).reset_index(drop=True)
yref=Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months) if which=='SEER' else Surv.from_arrays(d.os_status.astype(bool),d.survival_months)
names=list(P); n=len(d)
def mf(idx):
    dd=d.iloc[idx]; y=Surv.from_arrays(dd.os_status.astype(bool),dd.survival_months); r={}
    for k in names:
        lp,S=P[k]; c=concordance_index_censored(dd.os_status.astype(bool),dd.survival_months,lp[idx])[0]
        auc=cumulative_dynamic_auc(yref,y,lp[idx],TIMES)[0]; _,br=brier_score(yref,y,S[idx],TIMES); r[k]=np.r_[c,auc,br]
    return r
full=mf(np.arange(n)); pickle.dump(full,open(f'full2_{which}.pkl','wb'))
rng=np.random.default_rng(seed); boots=[]; t0=time.time()
for b in range(B):
    try: boots.append(mf(rng.integers(0,n,n)))
    except Exception as ex: continue
    if b%25==0: print(b,round(time.time()-t0),flush=True)
pickle.dump(boots,open(f'boot2_{which}_{seed}.pkl','wb')); print('done',len(boots),flush=True)
