import sys,warnings,pickle;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
from mlmodels import *
from lifelines import CoxPHFitter
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
cols=list(X(tr_).columns); mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_); t=tr_.survival_months.values; e=tr_.os_status.values
ytr=Surv.from_arrays(tr_.os_status.astype(bool),t)
# fit
cox=fit_cox(X(tr_),tr_)
gb=GBCox().fit(Xtr,t,e); ds=DeepSurv(seed=42).fit(Xtr,t,e); rsf=pickle.load(open('rsf_model.pkl','rb'))
def breslow_S(lp_tr,lp_d):
    """Breslow baseline from training data (frozen), survival at TIMES for new lp."""
    o=np.argsort(t); tt=t[o]; ee=e[o]; w=np.exp(lp_tr-lp_tr.max())[o]; rev=np.cumsum(w[::-1])[::-1]
    H=[]; 
    for T0 in TIMES:
        m=(tt<=T0)&(ee==1); H.append(np.sum(1.0/rev[m]*1.0)/np.exp(0)) if m.any() else H.append(0.0)
    H=np.array(H)*np.exp(-lp_tr.max())   # undo shift: H0(t)=sum d/ sum exp(lp) = (1/rev)*exp(-shift)... 
    return np.exp(-np.outer(np.exp(lp_d),H))
preds={}
for nm,d,Xd in [('SEER',te_,Xte),('TCGA',tc_,Xtc)]:
    P={}
    lp=cox.predict(X(d).reindex(columns=cols,fill_value=0.0)); P['Cox']=(lp,surv_at(cox,X(d).reindex(columns=cols,fill_value=0.0))[0])
    for k,m in [('Gradient-boosted',gb),('DeepSurv',ds)]:
        P[k]=(m.predict(Xd),breslow_S(m.predict(Xtr),m.predict(Xd)))
    fn=rsf.predict_survival_function(Xd,return_array=True); tt=rsf.unique_times_
    S=np.column_stack([fn[:,np.searchsorted(tt,y,side='right')-1] for y in TIMES]); P['RSF']=(-fn[:,-1],S)
    preds[nm]=P
pickle.dump(preds,open('preds_rebuilt.pkl','wb'))
rows=[]
for nm,d,yref in [('SEER',te_,ytr),('TCGA',tc_,Surv.from_arrays(tc_.os_status.astype(bool),tc_.survival_months))]:
    for k,(lp,S) in preds[nm].items():
        c,auc,br=eval_set(lp,S,d,yref)
        dd=pd.DataFrame({'T':d.survival_months.values,'E':d.os_status.values,'lp':lp}); sl=CoxPHFitter().fit(dd,'T','E').params_['lp']
        rows.append([nm,k,round(c,3),*np.round(auc,3),round(br[2],3),round(sl,3),round(S[:,2].min(),3),round(S[:,2].max(),3)])
R=pd.DataFrame(rows,columns=['cohort','model','C','AUC1y','AUC3y','AUC5y','Brier5y','cal.slope(Cox refit of lp)','S5y min','S5y max']); print(R.to_string(index=False))
R.to_csv('table3_rebuilt.csv',index=False)
