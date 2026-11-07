import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
from sksurv.metrics import concordance_index_censored
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
cols=list(X(tr_).columns); mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_); t=tr_.survival_months.values; e=tr_.os_status.values
C=lambda d,r:concordance_index_censored(d.os_status.astype(bool),d.survival_months,r)[0]
print('GB subsample=0.8:')
for sd in (42,1,2):
    g=GBCox(subsample=0.8,seed=sd).fit(Xtr,t,e); print(' seed',sd,'SEER',round(C(te_,g.predict(Xte)),4),'TCGA',round(C(tc_,g.predict(Xtc)),4))
print('DeepSurv relu seeds:')
for sd in (42,1,2,3):
    d=DeepSurv(seed=sd).fit(Xtr,t,e); print(' seed',sd,'SEER',round(C(te_,d.predict(Xte)),4),'TCGA',round(C(tc_,d.predict(Xtc)),4))
