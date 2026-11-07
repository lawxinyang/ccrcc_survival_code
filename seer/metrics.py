import sys;sys.path.insert(0,'.')
from common import *
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, brier_score
from sksurv.util import Surv
TIMES=np.array([12.,36.,60.])
def fit_cox(Xtr,tr_):
    return CoxPHSurvivalAnalysis(alpha=0,ties='efron').fit(Xtr,Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months))
def surv_at(model,Xd,times=TIMES):
    lp=model.predict(Xd); H=model.cum_baseline_hazard_
    H0=np.array([H(t) for t in times]); return np.exp(-np.outer(np.exp(lp),H0)),lp   # n x len(times)
def eval_set(lp,S,d,y_ref,times=TIMES):
    y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months)
    c=concordance_index_censored(d.os_status.astype(bool),d.survival_months,lp)[0]
    auc=cumulative_dynamic_auc(y_ref,y,lp,times)[0]
    _,br=brier_score(y_ref,y,S,times)
    return c,auc,br
