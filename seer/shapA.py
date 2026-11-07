import sys,pickle,warnings,time;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
from shapx import *
M=pickle.load(open('models_rebuilt.pkl','rb')); cols=M['cols']
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_)
idx=pickle.load(open('shap_out.pkl','rb'))['idx']; grp=groups(cols); res={}; t0=time.time()
for r in range(5):
    bg=Xtr[np.random.default_rng(1000+r).choice(len(Xtr),50,replace=False)]
    for nm,f in [('Gradient-boosted',M['gb'].predict),('DeepSurv',M['ds'].predict)]:
        for coh,Xe in [('SEER',Xte[idx]),('TCGA',Xtc)]:
            phi,_,_=exact_shap(f,Xe,bg,grp); res[(r,nm,coh)]=np.abs(phi).mean(0)
    print('rep',r,round(time.time()-t0),flush=True)
pickle.dump(res,open('shapA.pkl','wb'))
