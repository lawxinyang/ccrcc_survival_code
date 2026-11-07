import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from lifelines import CoxPHFitter, KaplanMeierFitter
P=pickle.load(open('preds_all.pkl','rb'))
H=[12,36,60]; names=['Cox, linear age','Cox, spline age','Gradient-boosted','DeepSurv','RSF']
rows=[]; dec=[]
for which in ['SEER','TCGA']:
    d=(prep(te) if which=='SEER' else prep(tc.dropna(subset=['M_stage']))).reset_index(drop=True)
    for k in names:
        lp,S=P[which][k]
        for j,h in enumerate(H):
            T=np.minimum(d.survival_months.values,h); E=((d.survival_months.values<=h)&(d.os_status.values==1)).astype(int)
            s=np.clip(S[:,j],1e-6,1-1e-6); risk=1-s; cll=np.log(-np.log(s))
            km=KaplanMeierFitter().fit(T,E); obs=1-km.predict(h); ci=km.confidence_interval_.loc[km.confidence_interval_.index<=h].iloc[-1]
            obs_lo,obs_hi=1-ci.iloc[1],1-ci.iloc[0]
            f=CoxPHFitter().fit(pd.DataFrame({'T':T,'E':E,'cll':cll}),'T','E'); b=f.params_['cll']; lo,hi=f.confidence_intervals_.loc['cll'].values
            rows.append([which,k,h//12,round(risk.mean(),4),round(obs,4),round(obs_lo,4),round(obs_hi,4),round(obs-risk.mean(),4),round(obs/risk.mean(),3),round(b,3),round(lo,3),round(hi,3),round(risk.min(),3),round(risk.max(),3)])
            g=pd.qcut(risk,10,duplicates='drop',labels=False)
            for q in sorted(set(g)):
                m=g==q; kk=KaplanMeierFitter().fit(T[m],E[m]); dec.append([which,k,h//12,q+1,int(m.sum()),round(risk[m].mean(),4),round(1-kk.predict(h),4)])
C=pd.DataFrame(rows,columns=['cohort','model','year','mean_pred_risk','obs_KM_risk','obs_lo','obs_hi','obs_minus_pred','O/E','cll_slope','slope_lo','slope_hi','pred_min','pred_max'])
D=pd.DataFrame(dec,columns=['cohort','model','year','decile','n','mean_pred_risk','obs_KM_risk'])
C.to_csv('calib_summary.csv',index=False); D.to_csv('calib_deciles.csv',index=False)
pd.set_option('display.width',250); print(C.drop(columns=['obs_lo','obs_hi']).to_string(index=False))
