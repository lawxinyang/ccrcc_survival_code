import sys,time,warnings,pickle;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
cols=[c for c in X(tr_).columns]
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
t0=time.time()
rsf=RandomSurvivalForest(n_estimators=105,max_depth=5,min_samples_leaf=50,max_features='sqrt',n_jobs=1,random_state=42).fit(mk(tr_),Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months))
C=lambda d,r:concordance_index_censored(d.os_status.astype(bool),d.survival_months,r)[0]
print('RSF fit sec',round(time.time()-t0),flush=True)
print('RSF train C',round(C(tr_,rsf.predict(mk(tr_))),4),'SEER test',round(C(te_,rsf.predict(mk(te_))),4),'TCGA',round(C(tc_,rsf.predict(mk(tc_))),4),flush=True)
pickle.dump(rsf,open('rsf_model.pkl','wb'))
