import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from scipy.stats import beta
from sksurv.metrics import concordance_index_censored
P=pickle.load(open('preds_all.pkl','rb'));rng=np.random.default_rng(3);rows=[]
def cp(k,n):
    return (0.0 if k==0 else beta.ppf(.025,k,n-k+1),1.0 if k==n else beta.ppf(.975,k+1,n-k))
for coh,d in [('TCGA-KIRC',prep(tc.dropna(subset=['M_stage']))),('SEER test',prep(te))]:
    pk='TCGA' if coh.startswith('TCGA') else 'SEER'
    for var in ['race','sex']:
        for lv in d[var].unique():
            m=(d[var]==lv).values;n=int(m.sum());k=int(d.os_status[m].sum());lo,hi=cp(k,n)
            r=[coh,var,lv,n,k,f'{100*k/n:.1f} ({100*lo:.1f}-{100*hi:.1f})']
            for mdl in ['Gradient-boosted','Cox, spline age']:
                lp=P[pk][mdl][0][m];tt=d.survival_months.values[m];ee=d.os_status.values[m].astype(bool)
                if k<10 or (n-k)<10: r.append('not estimable (<10 deaths or <10 survivors)');continue
                c=concordance_index_censored(ee,tt,lp)[0];bs=[]
                for _ in range(1000):
                    i=rng.integers(0,n,n)
                    if ee[i].sum()>0: bs.append(concordance_index_censored(ee[i],tt[i],lp[i])[0])
                r.append(f'{c:.3f} ({np.percentile(bs,2.5):.3f}-{np.percentile(bs,97.5):.3f})')
            rows.append(r)
R=pd.DataFrame(rows,columns=['cohort','variable','level','n','deaths','crude_death_%_(Clopper-Pearson 95% CI)','C-index GB (95% CI)','C-index Cox spline (95% CI)'])
pd.set_option('display.width',250);pd.set_option('display.max_colwidth',60);print(R.to_string(index=False));R.to_csv('subgroup_metrics.csv',index=False)
