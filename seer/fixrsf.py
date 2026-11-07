import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
from sksurv.metrics import concordance_index_ipcw
O='/mnt/user-data/outputs/'
M=pickle.load(open('models_rebuilt.pkl','rb'));cols=M['cols'];rsf=pickle.load(open('rsf_model.pkl','rb'))
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
P=pickle.load(open('preds_all.pkl','rb'));trp=prep(tr);ytr=Surv.from_arrays(trp.os_status.astype(bool),trp.survival_months)
# (a) Table3_all_horizons_CI + final_table3_ci: replace RSF rows
R=pd.read_csv('rsf_revised_ci.csv')
T=pd.read_csv('Table3_all_horizons_CI.csv')
for _,r in R.iterrows():
    m=(T.cohort==r.cohort)&(T.model=='RSF')
    for c in ['C-index','AUC 1y','AUC 3y','AUC 5y','Brier 1y','Brier 3y','Brier 5y']: T.loc[m,c]=r[c]
T.to_csv('Table3_all_horizons_CI.csv',index=False);T.to_csv(O+'Table3_all_horizons_CI.csv',index=False)
# (b) Rebuilt_models_table_with_CI.csv (columns C, AUC1y,... Brier5y) and pairwise
A=pd.read_csv('final_table3_ci.csv')
for _,r in R.iterrows():
    m=(A.cohort==r.cohort)&(A.model=='RSF')
    for c_old,c_new in [('C','C-index'),('AUC1y','AUC 1y'),('AUC3y','AUC 3y'),('AUC5y','AUC 5y'),('Brier5y','Brier 5y')]: A.loc[m,c_old]=r[c_new]
A.to_csv('final_table3_ci.csv',index=False);A.to_csv(O+'Rebuilt_models_table_with_CI.csv',index=False)
PW=pd.read_csv('final_pairwise.csv');RP=pd.read_csv('rsf_revised_pairwise.csv')
PW=PW[~PW.comparison.str.startswith('RSF')]
RP=RP[RP.comparison.str.contains('Cox, spline age|Gradient|DeepSurv|AJCC')].copy();RP['comparison']=RP.comparison.str.replace(' (revised scoring)','',regex=False)
rn={'C-index':'C','C-index P':'C P'};RP=RP.rename(columns=rn)[['cohort','comparison','C','C P','AUC 5y','AUC 5y P','Brier 5y','Brier 5y P']]
PW=pd.concat([PW,RP],ignore_index=True);PW.to_csv('final_pairwise.csv',index=False);PW.to_csv(O+'Rebuilt_models_pairwise_differences.csv',index=False)
# (c) R111 censoring sensitivity RSF row
tcd=prep(tc.dropna(subset=['M_stage']));ytc=Surv.from_arrays(tcd.os_status.astype(bool),tcd.survival_months)
native=rsf.predict(mk(tcd));S=P['TCGA']['RSF'][1]
aown=[cumulative_dynamic_auc(ytc,ytc,1-S[:,j],np.array([h]))[0][0] for j,h in enumerate(TIMES)];asee=[cumulative_dynamic_auc(ytr,ytc,1-S[:,j],np.array([h]))[0][0] for j,h in enumerate(TIMES)]
c36=concordance_index_ipcw(ytc,ytc,native,tau=36)[0];c60=concordance_index_ipcw(ytc,ytc,native,tau=60)[0]
C=pd.read_csv('R111_TCGA_censoring_sensitivity.csv');m=C.model=='RSF'
for k,v in zip(['AUC12_own','AUC36_own','AUC60_own'],aown): C.loc[m,k]=round(v,3)
for k,v in zip(['AUC12_SEERref','AUC36_SEERref','AUC60_SEERref'],asee): C.loc[m,k]=round(v,3)
C.loc[m,'UnoC_tau36']=round(c36,3);C.loc[m,'UnoC_tau60']=round(c60,3);C.to_csv('R111_TCGA_censoring_sensitivity.csv',index=False);C.to_csv(O+'R111_TCGA_censoring_sensitivity.csv',index=False);print(C[m].T.to_string())
# (d) period_performance RSF rows
Dd='/mnt/user-data/attachments/ca62109b-a916-5cd4-8db3-c8244fc66b01/'
cl=pd.read_csv(Dd+'aaebbd20-seer_kirc_cleaned.csv');raw=pd.read_csv(Dd+'857d2ac9-seer_kirc_database.csv');cl['p']=(raw['Derived AJCC T, 6th ed (2004-2015)'].astype(str).str.contains('Blank')).values.astype(int)
k=['age','sex','race','laterality','grade','T_stage','N_stage','M_stage','stage_group','survival_months','os_status','css_status'];key=lambda d:d[k].astype(str).agg('|'.join,axis=1);cl['key']=key(cl);g=cl.groupby('key').p.mean()
def tag(d):
    m=d.copy();m['period']=key(m).map(g).values;m['period']=np.where(m.period==1,'2016-2017',np.where(m.period==0,'2004-2015','ambiguous'));return m
trq,teq=prep(tag(tr)),prep(tag(te));Xte=mk(prep(te));nat=rsf.predict(Xte);Ste=P['SEER']['RSF'][1];PP=pd.read_csv('period_performance.csv')
for pname in ['2004-2015','2016-2017']:
    msk=(teq.period==pname).values;d=teq[msk];ref=Surv.from_arrays(trq[trq.period==pname].os_status.astype(bool),trq[trq.period==pname].survival_months);y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months)
    c=concordance_index_ipcw(ref,y,nat[msk],tau=60)[0];a=[cumulative_dynamic_auc(ref,y,1-Ste[msk][:,j],np.array([h]))[0][0] for j,h in enumerate(TIMES)]
    mm=(PP['test subset']=='test '+pname)&(PP.model.str.startswith('RSF'));PP.loc[mm,['UnoC_tau60','AUC1y','AUC3y','AUC5y']]=[round(c,3)]+[round(x,3) for x in a]
PP.to_csv('period_performance.csv',index=False);PP.to_csv(O+'period_performance.csv',index=False);print(PP[PP.model.str.startswith('RSF')].to_string(index=False))
