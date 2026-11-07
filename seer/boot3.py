import sys,time,pickle;sys.path.insert(0,'.')
from metrics import *
which,B,seed=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
M=pickle.load(open('models_rebuilt.pkl','rb'));cols=M['cols'];rsf=pickle.load(open('rsf_model.pkl','rb'))
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
P=pickle.load(open('preds_all.pkl','rb'))[which]
tr_=prep(tr);d=(prep(te) if which=='SEER' else prep(tc.dropna(subset=['M_stage']))).reset_index(drop=True)
native=rsf.predict(mk(d));S=P['RSF'][1]   # S at 12/36/60 from the same RSF
yref=Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months) if which=='SEER' else Surv.from_arrays(d.os_status.astype(bool),d.survival_months)
n=len(d)
def mf(idx):
    dd=d.iloc[idx];y=Surv.from_arrays(dd.os_status.astype(bool),dd.survival_months)
    c=concordance_index_censored(dd.os_status.astype(bool),dd.survival_months,native[idx])[0]
    auc=[cumulative_dynamic_auc(yref,y,1-S[idx,j],np.array([h]))[0][0] for j,h in enumerate(TIMES)]
    _,br=brier_score(yref,y,S[idx],TIMES);return np.r_[c,auc,br]
full=mf(np.arange(n));pickle.dump({'RSF (revised scoring)':full},open(f'full3_{which}.pkl','wb'));print('full',np.round(full,4),flush=True)
rng=np.random.default_rng(seed);boots=[];t0=time.time()
for b in range(B):
    boots.append({'RSF (revised scoring)':mf(rng.integers(0,n,n))})
    if b%25==0: print(b,round(time.time()-t0),flush=True)
pickle.dump(boots,open(f'boot3_{which}_{seed}.pkl','wb'));print('done',len(boots),flush=True)
