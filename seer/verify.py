import sys,pickle,glob,re,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
from lifelines.utils import concordance_index
O='/mnt/user-data/outputs/';ok=[];bad=[]
def chk(name,cond,detail=''): (ok if cond else bad).append(f'{name} {detail}')
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
# 1 cohort sizes and events
chk('SEER train/test n',len(tr_)==61546 and len(te_)==26378);chk('TCGA n',len(tc_)==510)
chk('deaths',(int(tr_.os_status.sum()),int(te_.os_status.sum()),int(tc_.os_status.sum()))==(28091,12040,170))
# 2 independent recompute: Cox linear C on SEER test and TCGA; AJCC stage only
P=pickle.load(open('preds_all.pkl','rb'))
for coh,d in [('SEER',te_),('TCGA',tc_)]:
    for k in ['Cox, linear age','Cox, spline age','Gradient-boosted','DeepSurv','AJCC stage only']:
        c=concordance_index(d.survival_months,-P[coh][k][0],d.os_status);t=pd.read_csv('Table3_all_horizons_CI.csv');r=t[(t.cohort==coh)&(t.model==k)].iloc[0]['C-index']
        chk(f'C-index {coh} {k}',abs(c-float(r.split()[0]))<0.0006,f'recomputed {c:.4f} vs table {r.split()[0]}')
# 3 pairwise = difference of point estimates in Table 3 file
F=pd.read_csv('Table3_all_horizons_CI.csv');PW=pd.read_csv('final_pairwise.csv')
g=lambda coh,m,c:float(F[(F.cohort==coh)&(F.model==m)][c].iloc[0].split()[0])
for _,r in PW.iterrows():
    a,b=r.comparison.split(' minus ');a={'Cox, spline age':'Cox, spline age'}.get(a,a)
    d=float(r['C'].split()[0]);diff=g(r.cohort,a,'C-index')-g(r.cohort,b,'C-index');chk(f'pairwise C {r.cohort} {r.comparison}',abs(d-diff)<0.0016,f'{d:+.4f} vs rounded-table {diff:+.4f}')
# 4 period partition sums
pd_=pd.read_csv('period_descriptive.csv');chk('period train sum+588',pd_[pd_.set=='train'].n.sum()+588==61546);chk('period test sum+245',pd_[pd_.set=='test'].n.sum()+245==26378)
# 5 subgroup sums
sg=pd.read_csv('subgroup_metrics.csv')
for coh,N in [('TCGA-KIRC',510),('SEER test',26378)]:
    for v in ['race','sex']: chk(f'subgroup {coh} {v} sums',sg[(sg.cohort==coh)&(sg.variable==v)].n.sum()==N)
# 6 numbers at risk partition
nr=pd.read_csv('R111_numbers_at_risk.csv')
for _,r in nr.iterrows():
    N={'SEER test':26378,'TCGA-KIRC':510,'SEER training':61546}[r.cohort];chk(f'at-risk partition {r.cohort} {r.horizon_months}',r.at_risk_beyond+r.cum_events+r.cum_censored==N)
# 7 DCA treat-all zero crossing TCGA 5y vs KM risk
d=pd.read_csv(O+'DCA_all_strategies.csv');t=d[(d.cohort=='TCGA')&(d.year==5)&(d.strategy=='Treat all')].sort_values('threshold');cross=t[t.net_benefit<0].threshold.min();chk('DCA TCGA treat-all crosses zero near KM 5y risk 0.374',0.35<=cross<=0.40,f'first negative at {cross}')
# 8 RSF consistency across files (revised vs old scoring)
for f,col in [('Table3_all_horizons_CI.csv','C-index'),('final_table3_ci.csv','C'),('R111_TCGA_censoring_sensitivity.csv',None),('period_performance.csv',None)]:
    pass
print('OK',len(ok),'BAD',len(bad));[print('  BAD:',b) for b in bad]
