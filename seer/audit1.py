import sys;sys.path.insert(0,'.')
from common import *
from scipy import stats
from lifelines import KaplanMeierFitter
tc2=tc.dropna(subset=['M_stage']).copy()
se=pd.concat([tr,te]); 
def pct(n,d): return f'{n:,} ({100*n/d:.1f})'
def block(name,d):
    n=len(d); print(f'\n[{name}] n={n:,}')
    print(' age mean(SD)',round(d.age.mean(),1),round(d.age.std(),1))
    print(' male',pct((d.sex=='Male').sum(),n))
    for v,lv in [('race',['White','Black','Asian','Other/Unknown']),('grade',['G1','G2','G3','G4','GX']),('T_stage',['T1','T2','T3','T4']),('N_stage',['N0','N+','NX']),('M_stage',['M0','M1','MX'])]:
        print(' ',v,{l:pct((d[v]==l).sum(),n) for l in lv})
    print('  T0/TX',pct(d.T_stage.isin(['T0','TX']).sum(),n))
    print('  grade3-4',pct(d.grade.isin(['G3','G4']).sum(),n))
    print('  deaths',pct(int(d.os_status.sum()),n))
    k=KaplanMeierFitter().fit(d.survival_months,1-d.os_status); print('  median FU reverse-KM',round(k.median_survival_time_,1),' median of raw times',round(d.survival_months.median(),1))
for nm,d in [('train',tr),('test',te),('SEER all',se),('TCGA',tc2)]: block(nm,d)
# tests
def chi(a,b,v):
    ct=pd.crosstab(pd.concat([a[v],b[v]]),np.r_[np.zeros(len(a)),np.ones(len(b))]); return stats.chi2_contingency(ct,correction=False)[1]
print('\nP values train vs test:', {v:round(chi(tr,te,v),3) for v in ['sex','race','grade','T_stage','N_stage','M_stage']}, 'age', round(stats.ttest_ind(tr.age,te.age,equal_var=False).pvalue,3))
print('P SEER vs TCGA:',{v:float(f'{chi(se,tc2,v):.2e}') for v in ['sex','race','grade','T_stage','N_stage','M_stage']},'age',round(stats.ttest_ind(se.age,tc2.age,equal_var=False).pvalue,3))
print('SEER mean age',round(se.age.mean(),1),'sex male %',round(100*(se.sex=='Male').mean(),1),'TCGA male %',round(100*(tc2.sex=='Male').mean(),1))
print('zero-month: train',(tr.survival_months==0).sum(),'test',(te.survival_months==0).sum(),'SEER total',(se.survival_months==0).sum(),'TCGA',(tc2.survival_months==0).sum())
