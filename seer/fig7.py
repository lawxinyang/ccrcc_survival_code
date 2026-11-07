import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
import numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from shapx import VARS
plt.rcParams.update({'font.family':'Liberation Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
LAB={'age':'Age','sex':'Sex','race':'Race','grade':'Grade','T_stage':'T stage','N_stage':'N stage','M_stage':'M stage'}
so=pickle.load(open('shap_out.pkl','rb')); out,idx=so['out'],so['idx']; M=pickle.load(open('models_rebuilt.pkl','rb')); cols=M['cols']
R=pd.read_csv('shap_importance_ci.csv')
te_,tc_=prep(te),prep(tc.dropna(subset=['M_stage']))
age={'SEER':te_.age.values[idx],'TCGA':tc_.age.values}
fig,ax=plt.subplots(2,2,figsize=(10,8))
for j,coh in enumerate(['SEER','TCGA']):
    a=ax[0,j]; d=R[(R.model=='Gradient-boosted')&(R.cohort==coh)].sort_values('mean_abs_SHAP')
    a.barh([LAB[v] for v in d.variable],d.mean_abs_SHAP,xerr=[d.mean_abs_SHAP-d.lo,d.hi-d.mean_abs_SHAP],color='#4a4a4a',ecolor='black',capsize=2)
    a.set_xlabel('Mean |SHAP| (log relative hazard)'); a.set_title(f"{'AB'[j]}. {'SEER test subsample (n = 1,000)' if coh=='SEER' else 'TCGA-KIRC (n = 510)'}",loc='left')
    b=ax[1,j]
    for nm,c,mk in [('Cox','#1f77b4','o'),('Gradient-boosted','#d62728','s'),('DeepSurv','#2ca02c','^')]:
        phi=out[(nm,coh)][0][:,0]; b.scatter(age[coh],phi,s=6 if coh=='SEER' else 10,alpha=.35,color=c,marker=mk,label=nm,linewidths=0)
    b.axhline(0,color='black',lw=.6); b.set_xlabel('Age at diagnosis (years)'); b.set_ylabel('SHAP value for age (log relative hazard)')
    b.set_title(f"{'CD'[j]}. Age dependence, {coh}",loc='left'); b.legend(frameon=False,markerscale=2)
plt.tight_layout(); plt.savefig('Figure7_SHAP_rebuilt.png',dpi=300); plt.savefig('Figure7_SHAP_rebuilt.pdf')
# rank agreement table (R1 m13)
rows=[]
for coh in ['SEER','TCGA']:
    P=R[R.cohort==coh].pivot(index='variable',columns='model',values='mean_abs_SHAP')
    from scipy.stats import spearmanr
    for a_,b_ in [('Gradient-boosted','DeepSurv'),('Gradient-boosted','Cox'),('DeepSurv','Cox')]:
        rho=spearmanr(P[a_],P[b_]).statistic; rows.append([coh,a_,b_,round(rho,3)])
print(pd.DataFrame(rows,columns=['cohort','model_a','model_b','spearman']).to_string(index=False))
print(R[R.model=='Gradient-boosted'].sort_values(['cohort','rank'])[['cohort','variable','mean_abs_SHAP','lo','hi','rank']].to_string(index=False))
