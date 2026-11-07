import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats
tr_=prep(tr); M=X(tr_); d=M.copy(); d['T']=tr_.survival_months.values; d['E']=tr_.os_status.values
f=CoxPHFitter().fit(d,'T','E')
rows=[]
for tt in ['rank','km']:
    r=proportional_hazard_test(f,d,time_transform=tt)
    s=r.summary; s.columns=[f'{c}_{tt}' for c in s.columns]; rows.append(s)
S=pd.concat(rows,axis=1)
# effect size: Spearman rho between scaled Schoenfeld residual and time-rank (events only)
res=f.compute_residuals(d,'scaled_schoenfeld'); ev=res.index
tm=d.loc[ev,'T']; rho={c:stats.spearmanr(res[c],tm)[0] for c in res.columns}
S['rho_time']=pd.Series(rho)
S=S[['test_statistic_rank','p_rank','test_statistic_km','p_km','rho_time']]
print(S.round(4).to_string())
# global test: sum of per-covariate not valid; use lifelines multivariate? report via residual correlation
S.to_pickle('ph_table.pkl')
# band-specific HRs: administrative censoring at band end, entry at band start
bands=[(0,12),(12,36),(36,60),(60,1e9)]
out={}
for lo,hi in bands:
    b=d[d['T']>lo].copy(); b['entry']=lo
    e=(b['T']<=hi)&(b['E']==1); b['Eb']=e.astype(int); b['Tb']=np.minimum(b['T'],hi)
    bb=b.drop(columns=['T','E']).rename(columns={'Tb':'T','Eb':'E'})
    fm=CoxPHFitter().fit(bb.drop(columns=['entry']),'T','E',entry_col=None) if False else CoxPHFitter().fit(bb,'T','E',entry_col='entry')
    out[f'{lo}-{int(hi) if hi<1e8 else "+"}']=np.exp(fm.params_); print('band',lo,hi,'n',len(bb),'events',int(bb.E.sum()),flush=True)
H=pd.DataFrame(out); H['overall']=np.exp(f.params_); H.to_pickle('ph_bands.pkl'); print(H.round(2).to_string())
