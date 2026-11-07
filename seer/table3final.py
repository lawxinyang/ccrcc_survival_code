import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
from sksurv.nonparametric import kaplan_meier_estimator
T3=pd.read_csv('Table3_all_horizons_CI.csv');full=pd.read_csv('final_table3_ci.csv')
cal=pd.read_csv('/mnt/user-data/outputs/Calibration_summary.csv');lr=pd.read_csv('R111_risk_tertiles.csv')
trp=prep(tr);ytr=Surv.from_arrays(trp.os_status.astype(bool),trp.survival_months)
# null (KM-only) IPCW Brier
null={}
for coh,d,ref in [('SEER',prep(te),ytr),('TCGA',prep(tc.dropna(subset=['M_stage'])),None)]:
    y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months);r=ytr if ref is not None else y
    tt,s=kaplan_meier_estimator(r['event'],r['time']);S0=np.array([s[np.searchsorted(tt,h,side='right')-1] for h in TIMES])
    null[coh]=[round(x,3) for x in brier_score(r,y,np.tile(S0,(len(d),1)),TIMES)[1]]
print('null-model Brier (1,3,5 y):',null)
order=['AJCC stage only','Cox, linear age','Cox, spline age','Cox, spline + interactions','RSF','Gradient-boosted','DeepSurv']
rows=[]
for coh in ['SEER','TCGA']:
    for m in order:
        r=T3[(T3.cohort==coh)&(T3.model==m)].iloc[0];row={'cohort':coh,'model':m}
        for c in ['C-index','AUC 1y','AUC 3y','AUC 5y','Brier 1y','Brier 3y','Brier 5y']: row[c]=r[c]
        c5=cal[(cal.cohort==coh)&(cal.model==m)&(cal.year==5)]
        if len(c5): c5=c5.iloc[0];row['5y cloglog calibration slope (95% CI)']=f"{c5.cll_slope:.2f} ({c5.slope_lo:.2f}-{c5.slope_hi:.2f})";row['5y mean predicted / KM observed risk']=f"{c5.mean_pred_risk:.3f} / {c5.obs_KM_risk:.3f}";row['5y O/E']=c5['O/E']
        rows.append(row)
R=pd.DataFrame(rows);R.to_csv('Table3_final_rebuilt.csv',index=False);pd.set_option('display.width',300);pd.set_option('display.max_columns',20);print(R.to_string(index=False))
pickle.dump(null,open('null_brier.pkl','wb'))
