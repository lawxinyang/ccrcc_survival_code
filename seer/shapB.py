import sys,pickle,warnings,time;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
from gbclip import GBCoxClip
from shapx import *
M=pickle.load(open('models_rebuilt.pkl','rb')); cols=M['cols']
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_); t=tr_.survival_months.values; e=tr_.os_status.values
idx=pickle.load(open('shap_out.pkl','rb'))['idx'][:300]; grp=groups(cols)
bg=Xtr[np.random.default_rng(42).choice(len(Xtr),50,replace=False)]; res={}; t0=time.time()
for r in range(10):
    ds=DeepSurv(seed=100+r).fit(Xtr,t,e)
    gb=None
    for k in range(5):
        b=np.random.default_rng(3000+10*r+k).choice(len(Xtr),int(.8*len(Xtr)),replace=False)
        try: gb=GBCox().fit(Xtr[b],t[b],e[b]); break
        except ValueError: print('GB divergence, retry',r,k,flush=True)
    if gb is None: continue
    for nm,f in [('Gradient-boosted (80% refit)',gb.predict),('DeepSurv (new seed)',ds.predict)]:
        for coh,Xe in [('SEER',Xte[idx]),('TCGA',Xtc)]:
            phi,_,_=exact_shap(f,Xe,bg,grp); res[(r,nm,coh)]=np.abs(phi).mean(0)
    print('refit',r,round(time.time()-t0),flush=True)
    pickle.dump(res,open('shapB.pkl','wb'))
