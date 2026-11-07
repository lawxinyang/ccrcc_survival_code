import sys;sys.path.insert(0,'.')
from metrics import *
tr_all=prep(tr)
KN=np.quantile(tr_all.age,[.05,.35,.65,.95])
def _dum(d,v,cols=None):
    z=pd.get_dummies(d[v]).astype(float); z.columns=[f'{v}_{c}' for c in z.columns]; return z
def fx_stage(d): return _dum(d,'stage_group').drop(columns=['stage_group_I'],errors='ignore')
def fx_tnm(d):
    o=X(d,('T_stage','N_stage','M_stage')); return o
def fx_lin(d): return X(d)
def fx_spl(d): return X_spline(d,KN)
def fx_ext(d):
    B=X_spline(d,KN)
    for g,pre in [('M','M_stage_'),('N','N_stage_'),('G','grade_')]:
        for c in [c for c in B.columns if c.startswith(pre)]: B[f'age*{c}']=B['age']*B[c]
    for c in ['T_stage_T2','T_stage_T3','T_stage_T4']: B[f'age*{c}']=B['age']*B[c]
    return B
def fx_spl_noN(d): return X_spline(d,KN,('age','sex','race','grade','T_stage','M_stage'))
def fx_spl_noRace(d): return X_spline(d,KN,('age','sex','grade','T_stage','N_stage','M_stage'))
def fx_spl_all8(d): return X_spline(d,KN,('age','sex','race','laterality','grade','T_stage','N_stage','M_stage'))
VARS1={'AJCC stage only':fx_stage,'T/N/M only (linear)':fx_tnm,'Cox, linear age (7 var)':fx_lin,'Cox, spline age (7 var)':fx_spl,'Cox, spline + interactions':fx_ext}
VARS2={'Cox, spline age, no N stage':fx_spl_noN,'Cox, spline age, no race':fx_spl_noRace,'Cox, spline age, all 8 variables':fx_spl_all8}
def fit_all(VARS):
    out={}
    for k,fx in VARS.items():
        Xtr=fx(tr_all); cols=list(Xtr.columns); out[k]=(fx,cols,fit_cox(Xtr,tr_all))
    return out
def predict(models,d):
    P={}
    for k,(fx,cols,m) in models.items():
        Xd=fx(d).reindex(columns=cols,fill_value=0.0); S,lp=surv_at(m,Xd); P[k]=(lp,S)
    return P
