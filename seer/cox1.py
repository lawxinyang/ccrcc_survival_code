import sys;sys.path.insert(0,'.')
from common import *
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
TIES=sys.argv[1]
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
print(len(tc_),tc_.os_status.sum(),tc_.survival_months.median())
Xtr=X(tr_);cols=list(Xtr.columns)
def mk(d): return X(d).reindex(columns=cols,fill_value=0.0)
ytr=Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months)
m=CoxPHSurvivalAnalysis(alpha=0,ties=TIES).fit(Xtr,ytr)
for n,d in [('SEER test',te_),('TCGA',tc_)]:
    r=m.predict(mk(d)); c=concordance_index_censored(d.os_status.astype(bool),d.survival_months,r)[0]
    print(n,'Cox linear-age C-index',round(c,4))
import pandas as pd
print(pd.Series(np.exp(m.coef_),index=cols).round(3).to_string())
