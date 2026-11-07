import sys,warnings,pickle;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
tr_=prep(tr); cols=[c for c in X(tr_).columns if c!='T_stage_T0']
M=X(tr_)[cols]; T=tr_.survival_months.values; E=tr_.os_status.values
bands=[(0,12),(12,36),(36,60)]; models=[]
for lo,hi in bands:
    a=T>lo; t=T[a]-lo; e=E[a]; w=hi-lo; e2=((t<=w)&(e==1)); t2=np.minimum(t,w)
    models.append((fit:=CoxPHSurvivalAnalysis(alpha=1e-5,ties='breslow',n_iter=200).fit(M[a],Surv.from_arrays(e2,t2)),w))
std=CoxPHSurvivalAnalysis(alpha=1e-5,ties='breslow',n_iter=200).fit(M,Surv.from_arrays(E.astype(bool),T))
def piecewise_S(d):
    Xd=X(d).reindex(columns=cols,fill_value=0.0); S=np.ones((len(d),3)); cum=np.ones(len(d))
    for k,(m,w) in enumerate(models):
        lp=m.predict(Xd); Sk=np.exp(-m.cum_baseline_hazard_(w)*np.exp(lp)); cum=cum*Sk; S[:,k]=cum
    return S
def std_S(d):
    Xd=X(d).reindex(columns=cols,fill_value=0.0); lp=std.predict(Xd); return np.exp(-np.outer(np.exp(lp),[std.cum_baseline_hazard_(t) for t in TIMES])),lp
ytr=Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months)
for nm,d,yref in [('SEER',prep(te),ytr),('TCGA',prep(tc.dropna(subset=['M_stage'])),None)]:
    y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months); yr=y if yref is None else yref
    Sp=piecewise_S(d); Ss,lp=std_S(d)
    for lab,S in [('standard Cox',Ss),('piecewise (time-varying)',Sp)]:
        auc=[cumulative_dynamic_auc(yr,y,1-S[:,i],[TIMES[i]])[0][0] for i in range(3)]
        _,br=brier_score(yr,y,S,TIMES)
        c=concordance_index_censored(d.os_status.astype(bool),d.survival_months,1-S[:,2])[0]
        print(nm,lab,'C(1-S60)',round(c,4),'AUC',np.round(auc,4),'Brier',np.round(br,4))
    # agreement in predicted 5y survival
    print(nm,'5-y predicted survival: mean abs diff',round(np.abs(Sp[:,2]-Ss[:,2]).mean(),4),'max',round(np.abs(Sp[:,2]-Ss[:,2]).max(),3),' corr',round(np.corrcoef(Sp[:,2],Ss[:,2])[0,1],4))
