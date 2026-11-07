import sys;sys.path.insert(0,'.')
from common import *
from lifelines import CoxPHFitter
from scipy import stats
tr_=prep(tr); kn=np.quantile(tr_.age,[.05,.35,.65,.95])
B=X_spline(tr_,kn)
def ll(M):
    d=M.copy(); d['T']=tr_.survival_months.values; d['E']=tr_.os_status.values
    return CoxPHFitter().fit(d,'T','E').log_likelihood_
base=ll(B); print('base spline LL',round(base,1))
def grp(prefix): return [c for c in B.columns if c.startswith(prefix)]
cands={'age x M stage':('age',grp('M_stage')),'age x N stage':('age',grp('N_stage')),'age x T stage':('age',grp('T_stage')),'age x grade':('age',grp('grade')),'age x sex':('age',grp('sex')),
 'sex x M stage':(grp('sex'),grp('M_stage')),'grade x M stage':(grp('grade'),grp('M_stage')),'T stage x M stage':(grp('T_stage'),grp('M_stage')),'T stage x N stage':(grp('T_stage'),grp('N_stage')),'grade x T stage':(grp('grade'),grp('T_stage')),'race x age':(grp('race'),'age')}
for nm,(a,b) in cands.items():
    a=[a] if isinstance(a,str) else a; b=[b] if isinstance(b,str) else b
    M=B.copy(); added=0
    for x in a:
        for y in b: M[f'{x}*{y}']=B[x]*B[y]; added+=1
    lr=2*(ll(M)-base); print(f'{nm:20s} df={added:2d}  LR chi2={lr:8.1f}  P={stats.chi2.sf(lr,added):.2e}')
