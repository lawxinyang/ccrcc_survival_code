import sys,time,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
from sksurv.metrics import concordance_index_censored
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
cols=list(X(tr_).columns)
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_)
C=lambda d,r:concordance_index_censored(d.os_status.astype(bool),d.survival_months,r)[0]
t0=time.time()
gb=GBCox().fit(Xtr,tr_.survival_months.values,tr_.os_status.values)
print('GB (no subsample) rounds=216 | train C',round(C(tr_,gb.train_F),4),'| SEER test',round(C(te_,gb.predict(Xte)),4),'| TCGA',round(C(tc_,gb.predict(Xtc)),4),'| sec',round(time.time()-t0))
for n in (100,150,216,300): pass
t0=time.time()
for act in ['relu','tanh']:
    ds=DeepSurv(act=act).fit(Xtr,tr_.survival_months.values,tr_.os_status.values)
    print('DeepSurv',act,'| train C',round(C(tr_,ds.predict(Xtr)),4),'| SEER test',round(C(te_,ds.predict(Xte)),4),'| TCGA',round(C(tc_,ds.predict(Xtc)),4),'| final loss',round(ds.hist[-1],4),'| sec',round(time.time()-t0))
