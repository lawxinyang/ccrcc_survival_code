import sys,time;sys.path.insert(0,'.')
from metrics import *
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
kn=np.quantile(tr_.age,[.05,.35,.65,.95]); print('knots',kn)
ytr=Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months)
yte=Surv.from_arrays(te_.os_status.astype(bool),te_.survival_months); ytc=Surv.from_arrays(tc_.os_status.astype(bool),tc_.survival_months)
print('TCGA age range',tc_.age.min(),tc_.age.max(),' train',tr_.age.min(),tr_.age.max())
res={}
for tag,fx in [('linear',lambda d:X(d)),('spline',lambda d:X_spline(d,kn))]:
    Xtr=fx(tr_); cols=list(Xtr.columns); m=fit_cox(Xtr,tr_)
    for nm,d,yref in [('SEER',te_,ytr),('TCGA',tc_,ytc)]:
        Xd=fx(d).reindex(columns=cols,fill_value=0.0); S,lp=surv_at(m,Xd)
        t0=time.time(); c,auc,br=eval_set(lp,S,d,yref); 
        res[(tag,nm)]=(c,auc,br); print(tag,nm,'C',round(c,4),'AUC12/36/60',np.round(auc,4),'Brier12/36/60',np.round(br,4),'time',round(time.time()-t0,1))
