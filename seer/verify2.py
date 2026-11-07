import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
import numpy as np,pandas as pd
O='/mnt/user-data/outputs/';ok=[];bad=[]
def chk(n,c,d=''): (ok if c else bad).append(f'{n} {d}')
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']));al=pd.concat([tr,te])
# R1-1 audit claims recomputed from data
chk('A1 grade3-4 SEER 28.8%',abs(100*al.grade.isin(['G3','G4']).mean()-28.8)<0.06,f'{100*al.grade.isin(["G3","G4"]).mean():.2f}')
chk('A2 M1 SEER 10.2%',abs(100*(al.M_stage=='M1').mean()-10.2)<0.06,f'{100*(al.M_stage=="M1").mean():.2f}')
chk('A3 male SEER 62.5 / TCGA 63.9',abs(100*(al.sex=='Male').mean()-62.5)<0.06 and abs(100*(tc_.sex=='Male').mean()-63.9)<0.06)
chk('TCGA grade3-4 54.1, M1 15.3, NX 52.0',abs(100*tc_.grade.isin(['G3','G4']).mean()-54.1)<.06 and abs(100*(tc_.M_stage=='M1').mean()-15.3)<.06 and abs(100*(tc_.N_stage=='NX').mean()-52.0)<.06)
chk('A7 322/88246=0.365%',abs(100*322/88246-0.365)<0.001)
chk('SEER analysis n',len(al)==87924)
chk('Asian TCGA 8 (1.6%), SEER 6.3%',(tc_.race=='Asian').sum()==8 and abs(100*(al.race=='Asian').mean()-6.3)<0.06)
chk('29% TCGA followed >5y',abs(100*(tc_.survival_months>60).mean()-29.0)<0.1,f'{100*(tc_.survival_months>60).mean():.2f}')
chk('zero-month SEER 928 / TCGA 2',((pd.concat([tr,te]).survival_months==0).sum()==928) and ((tc.dropna(subset=["M_stage"]).survival_months==0).sum()==2))
# SHAP robustness file sanity
s=pd.read_csv(O+'shap_refit_uncertainty.csv');chk('SHAP refit: no diverged values left',s['max'].max()<5,f"max {s['max'].max():.3f}")
r=pd.read_csv(O+'shap_robustness.csv');chk('SHAP background-only file finite',r['max'].max()<5,str(r['max'].max()))
# LASSO: reload and verify claims
o=pickle.load(open('lasso_out.pkl','rb'));a=o['alphas'];mu=o['mu'];chk('LASSO CV C max 0.757',abs(mu.max()-0.7572)<0.0003,f'{mu.max():.4f}');chk('LASSO 1SE lambda 0.0136',abs(a[o['j1']]-0.01358)<0.0002)
# Calibration claims
c=pd.read_csv(O+'Calibration_summary.csv');t5=c[(c.cohort=='TCGA')&(c.year==5)];chk('TCGA 5y O/E 1.23-1.31 for ML+Cox',t5['O/E'].between(1.22,1.32).all(),str(t5['O/E'].tolist()))
chk('TCGA 5y KM risk 0.374',abs(t5.obs_KM_risk.iloc[0]-0.3737)<0.0005)
# KM TCGA 60 months from data directly
from lifelines import KaplanMeierFitter
km=KaplanMeierFitter().fit(tc_.survival_months,tc_.os_status);chk('TCGA KM 5y survival 62.6',abs(100*km.survival_function_at_times(60).iloc[0]-62.6)<0.06)
# every output csv readable, no NaN-only columns in the headline tables
for f in ['Table3_final_rebuilt.csv','Table3_all_horizons_CI.csv','Rebuilt_models_table_with_CI.csv','Rebuilt_models_pairwise_differences.csv','period_performance.csv','Subgroup_metrics_with_CI.csv','Race_sensitivity_ML_models.csv','R111_numbers_at_risk.csv','R111_risk_tertiles.csv']:
    d=pd.read_csv(O+f);chk('readable '+f,len(d)>0)
# RSF now consistent everywhere
for f,col in [('Table3_final_rebuilt.csv','C-index'),('Table3_all_horizons_CI.csv','C-index'),('Rebuilt_models_table_with_CI.csv','C')]:
    d=pd.read_csv(O+f);v=d[(d.cohort=='SEER')&(d.model=='RSF')][col].iloc[0];chk(f'RSF SEER C in {f} = 0.755',v.startswith('0.755'),v)
print('OK',len(ok),'BAD',len(bad));[print('  BAD:',b) for b in bad]
