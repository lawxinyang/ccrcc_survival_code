"""Cox fitting, Breslow baseline (frozen from the training set), and evaluation metrics."""
import numpy as np, pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, brier_score
from sksurv.util import Surv
TIMES = np.array([12., 36., 60.])
def surv_obj(d): return Surv.from_arrays(d.os_status.astype(bool), d.survival_months)
def fit_cox(X, d): return CoxPHSurvivalAnalysis(alpha=0, ties='efron').fit(X, surv_obj(d))
def breslow_baseline(lp_train, t, e):
    """Breslow cumulative baseline hazard at each distinct event time: H0(t) = sum_{t_i <= t} d_i / sum_{j: t_j >= t_i} exp(lp_j)."""
    shift = lp_train.max(); w = np.exp(lp_train - shift)
    order = np.argsort(t, kind='stable'); ts, es, ws = t[order], e[order], w[order]
    rev = np.cumsum(ws[::-1])[::-1]
    ut, first = np.unique(ts, return_index=True)
    d_i = np.add.reduceat(es.astype(float), first)
    H = np.cumsum(d_i / rev[first]) * np.exp(-shift)
    return ut, H
def surv_from_baseline(lp, ut, H, times=TIMES):
    """Right-continuous step function, no interpolation."""
    idx = np.searchsorted(ut, times, side='right') - 1
    H0 = np.where(idx >= 0, H[np.clip(idx, 0, None)], 0.0)
    return np.exp(-np.outer(np.exp(lp), H0))
def evaluate(risk, S, d, ref_surv, times=TIMES, auc_risk=None):
    """C-index of `risk`; cumulative/dynamic AUC (Uno IPCW) and IPCW Brier from S (n x len(times)). ref_surv: data defining the censoring distribution."""
    y = surv_obj(d)
    c = concordance_index_censored(d.os_status.astype(bool), d.survival_months, risk)[0]
    auc = [cumulative_dynamic_auc(ref_surv, y, (1 - S[:, j]) if auc_risk is None else auc_risk[:, j], np.array([h]))[0][0] for j, h in enumerate(times)]
    br = brier_score(ref_surv, y, S, times)[1]
    return c, np.array(auc), np.asarray(br)
