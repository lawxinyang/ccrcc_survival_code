import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_ipcw
P=pickle.load(open('preds_all.pkl','rb'))
trp=prep(tr); D={'SEER test':prep(te),'TCGA-KIRC':prep(tc.dropna(subset=['M_stage'])),'SEER training':trp}
ytr=Surv.from_arrays(trp.os_status.astype(bool),trp.survival_months)
rows=[];fu=[]
for nm,d in D.items():
    t=d.survival_months.values;e=d.os_status.values.astype(bool);N=len(d)
    km=KaplanMeierFitter().fit(t,e); rk=KaplanMeierFitter().fit(t,~e)
    m=km.median_survival_time_; mci=median_survival_times(km.confidence_interval_)
    rm=rk.median_survival_time_; rci=median_survival_times(rk.confidence_interval_)
    fu.append([nm,N,int(e.sum()),round(np.median(t),1),round(rm,1),round(rci.iloc[0,0],1),round(rci.iloc[0,1],1),round(m,1) if np.isfinite(m) else 'not reached'])
    for T in (12,36,60):
        s=km.survival_function_at_times(T).iloc[0]; ci=km.confidence_interval_survival_function_
        lo=ci.loc[:T].iloc[-1,0];hi=ci.loc[:T].iloc[-1,1]
        rows.append([nm,T,int((t>T).sum()),int(((t<=T)&e).sum()),int(((t<=T)&~e).sum()),round(100*s,1),round(100*lo,1),round(100*hi,1),round(100*(1-s),1)])
A=pd.DataFrame(rows,columns=['cohort','horizon_months','at_risk_beyond','cum_events','cum_censored','KM_survival_%','lo','hi','KM_risk_%'])
F=pd.DataFrame(fu,columns=['cohort','N','deaths','median_observed_time','median_FU_reverseKM','lo','hi','median_OS_KM'])
pd.set_option('display.width',200);print(A.to_string(index=False));print(F.to_string(index=False))
A.to_csv('R111_numbers_at_risk.csv',index=False);F.to_csv('R111_followup.csv',index=False)
# censoring distribution / IPCW weights
cw=[]
for nm,d,ref in [('SEER test (reference: SEER training)',D['SEER test'],ytr),('TCGA-KIRC (own censoring)',D['TCGA-KIRC'],Surv.from_arrays(D['TCGA-KIRC'].os_status.astype(bool),D['TCGA-KIRC'].survival_months))]:
    tt,Gs=kaplan_meier_estimator(~ref['event'],ref['time'])
    G=lambda x:Gs[np.searchsorted(tt,x,side='right')-1] if x>=tt[0] else 1.0
    y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months); ev=y['time'][y['event']&(y['time']<=60)]
    w=1/np.array([G(x) for x in ev])
    cw.append([nm]+[round(G(T),3) for T in (12,36,60)]+[round(w.min(),2),round(np.median(w),2),round(np.percentile(w,99),2),round(w.max(),2),len(ev)])
C=pd.DataFrame(cw,columns=['cohort','G(12)','G(36)','G(60)','w_min','w_median','w_p99','w_max','n_events<=60']);print(C.to_string(index=False)); C.to_csv('R111_ipcw_weights.csv',index=False)
# censoring sensitivity for TCGA: reference distribution variants and truncation horizon
tcd=D['TCGA-KIRC'];ytc=Surv.from_arrays(tcd.os_status.astype(bool),tcd.survival_months);rows=[]
for mdl,(lp,S) in P['TCGA'].items():
    a_own=cumulative_dynamic_auc(ytc,ytc,lp,TIMES)[0]; a_seer=cumulative_dynamic_auc(ytr,ytc,lp,TIMES)[0]
    b_own=brier_score(ytc,ytc,S,TIMES)[1]; b_seer=brier_score(ytr,ytc,S,TIMES)[1]
    c36=concordance_index_ipcw(ytc,ytc,lp,tau=36)[0]; c60=concordance_index_ipcw(ytc,ytc,lp,tau=60)[0]
    rows.append([mdl]+[round(x,3) for x in list(a_own)+list(a_seer)+list(b_own)+list(b_seer)+[c36,c60]])
cols=['model']+[f'AUC{h}_own' for h in (12,36,60)]+[f'AUC{h}_SEERref' for h in (12,36,60)]+[f'Brier{h}_own' for h in (12,36,60)]+[f'Brier{h}_SEERref' for h in (12,36,60)]+['UnoC_tau36','UnoC_tau60']
S_=pd.DataFrame(rows,columns=cols);print(S_.T.to_string());S_.to_csv('R111_TCGA_censoring_sensitivity.csv',index=False)
