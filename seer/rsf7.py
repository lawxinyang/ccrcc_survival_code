import sys,warnings,pickle;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines.utils import concordance_index
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
cols=list(X(tr_).columns);mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
r=RandomSurvivalForest(n_estimators=105,max_depth=7,min_samples_leaf=50,max_features='sqrt',n_jobs=2,random_state=42).fit(mk(tr_),Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months))
for n,d in [('SEER',te_),('TCGA',tc_)]:
    print(n,round(concordance_index(d.survival_months,-r.predict(mk(d)),d.os_status),4))
