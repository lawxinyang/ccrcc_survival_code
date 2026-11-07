import sys,pickle,warnings,time;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
from metrics import fit_cox
from shapx import *
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
cols=list(X(tr_).columns); mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_); t=tr_.survival_months.values; e=tr_.os_status.values
cox=fit_cox(X(tr_),tr_); gb=GBCox().fit(Xtr,t,e); ds=DeepSurv(seed=42).fit(Xtr,t,e)
pickle.dump({'gb':gb,'ds':ds,'cols':cols},open('models_rebuilt.pkl','wb'))
fs={'Cox':lambda Z:Z@cox.coef_,'Gradient-boosted':gb.predict,'DeepSurv':ds.predict}
grp=groups(cols); rng=np.random.default_rng(42)
bg=Xtr[rng.choice(len(Xtr),50,replace=False)]
idx=np.random.default_rng(42).choice(len(Xte),1000,replace=False)
out={}; t0=time.time()
for nm,f in fs.items():
    for coh,Xe in [('SEER',Xte[idx]),('TCGA',Xtc)]:
        phi,base,vf=exact_shap(f,Xe,bg,grp); out[(nm,coh)]=(phi,base,vf)
        print(nm,coh,'additivity err',float(np.abs(phi.sum(1)+base-f(Xe)).max()),'| sec',round(time.time()-t0),flush=True)
pickle.dump({'out':out,'idx':idx,'bg':bg},open('shap_out.pkl','wb'))
rows=[]
for (nm,coh),(phi,_,_) in out.items(): rows.append([nm,coh]+list(np.round(np.abs(phi).mean(0),3)))
print(pd.DataFrame(rows,columns=['model','cohort']+VARS).to_string(index=False))
