import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from variants import *
from sksurv.metrics import concordance_index_ipcw
from lifelines import KaplanMeierFitter
Dd='/mnt/user-data/attachments/ca62109b-a916-5cd4-8db3-c8244fc66b01/'
cl=pd.read_csv(Dd+'aaebbd20-seer_kirc_cleaned.csv');raw=pd.read_csv(Dd+'857d2ac9-seer_kirc_database.csv')
cl['p1617']=(raw['Derived AJCC T, 6th ed (2004-2015)'].astype(str).str.contains('Blank')).values.astype(int)
k=['age','sex','race','laterality','grade','T_stage','N_stage','M_stage','stage_group','survival_months','os_status','css_status']
key=lambda d:d[k].astype(str).agg('|'.join,axis=1)
cl['key']=key(cl);g=cl.groupby('key').p1617.mean()
def tag(d):
    m=d.copy();m['period']=key(m).map(g).values;m['period']=np.where(m.period==1,'2016-2017',np.where(m.period==0,'2004-2015','ambiguous'));return m
trp,tep=tag(tr),tag(te);trp=prep(trp);tep=prep(tep)
print('ambiguous train/test',(trp.period=='ambiguous').sum(),(tep.period=='ambiguous').sum())
# 1 descriptive
rows=[]
for nm,d in [('train',trp),('test',tep)]:
    for p in ['2004-2015','2016-2017']:
        s=d[d.period==p];km=KaplanMeierFitter().fit(s.survival_months,s.os_status);rk=KaplanMeierFitter().fit(s.survival_months,1-s.os_status)
        rows.append([nm,p,len(s),int(s.os_status.sum()),round(s.age.mean(),1),round(100*(s.M_stage=='M1').mean(),1),round(100*s.grade.isin(['G3','G4']).mean(),1),round(100*(s.N_stage=='N+').mean(),1),round(100*(1-km.survival_function_at_times(60).iloc[0]),1),round(rk.median_survival_time_,1),s.survival_months.max()])
A=pd.DataFrame(rows,columns=['set','period','n','deaths','mean_age','M1_%','grade3-4_%','N+_%','KM_5y_mortality_%','median_FU_reverseKM','max_time']);print(A.to_string(index=False));A.to_csv('period_descriptive.csv',index=False)
# 2 coefficient comparison: Cox spline fitted separately by period (training)
cm=[]
for p in ['2004-2015','2016-2017']:
    s=trp[trp.period==p];Xs=fx_spl(s);m=fit_cox(Xs,s);cm.append(pd.Series(np.exp(m.coef_),index=Xs.columns,name=p))
Hc=pd.concat(cm,axis=1);Hc['ratio_2016-17_vs_2004-15']=Hc['2016-2017']/Hc['2004-2015'];print(Hc.round(3).to_string());Hc.round(4).to_csv('period_HR_by_period.csv')
# 3 models fitted on 2004-2015 only vs full, evaluated by period in test
res=[];TIM=np.array([12.,36.,60.])
def ev(lp,S,d,ref,label,model):
    y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months)
    c=concordance_index_ipcw(ref,y,lp,tau=60)[0];a=cumulative_dynamic_auc(ref,y,lp,TIM)[0];b=brier_score(ref,y,S,TIM)[1]
    return [label,model,len(d),int(d.os_status.sum()),round(c,3)]+[round(x,3) for x in a]+[round(x,3) for x in b]
T0=trp[trp.period=='2004-2015']
mA=fit_all({'Cox spline (fit 2004-2015 only)':fx_spl}) if False else None
fx=fx_spl;Xa=fx(T0);cols=list(Xa.columns);mold=fit_cox(Xa,T0);mfull=fit_all({'f':fx_spl})['f'][2]
for pname in ['2004-2015','2016-2017']:
    d=tep[tep.period==pname];ref=Surv.from_arrays(trp[trp.period==pname].os_status.astype(bool),trp[trp.period==pname].survival_months)
    Xd=fx(d).reindex(columns=cols,fill_value=0.0)
    for nm,m in [('Cox spline, fit on 2004-2015 training only',mold),('Cox spline, fit on full training set',mfull)]:
        S,lp=surv_at(m,Xd);res.append(ev(lp,S,d,ref,'test '+pname,nm))
P=pickle.load(open('preds_all.pkl','rb'))['SEER']
for pname in ['2004-2015','2016-2017']:
    msk=(tep.period==pname).values;d=tep[msk];ref=Surv.from_arrays(trp[trp.period==pname].os_status.astype(bool),trp[trp.period==pname].survival_months)
    for nm in ['AJCC stage only','Gradient-boosted','DeepSurv','RSF']:
        lp,S=P[nm];res.append(ev(lp[msk],S[msk],d,ref,'test '+pname,nm+' (full-train model)'))
R=pd.DataFrame(res,columns=['test subset','model','n','deaths','UnoC_tau60','AUC1y','AUC3y','AUC5y','Brier1y','Brier3y','Brier5y']);pd.set_option('display.width',220);print(R.to_string(index=False));R.to_csv('period_performance.csv',index=False)
