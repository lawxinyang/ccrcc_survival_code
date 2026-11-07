import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from lifelines import CoxPHFitter
tr_=prep(tr); M=X(tr_).drop(columns=['T_stage_T0']); d=M.copy(); d['T']=tr_.survival_months.values; d['E']=tr_.os_status.values
d=d[tr_['T_stage'].values!='T0'] if False else d
keep=(tr_['T_stage']!='T0').values; d=d[keep].reset_index(drop=True)
f=CoxPHFitter().fit(d,'T','E'); out={}
for lo,hi in [(0,12),(12,36),(36,60),(60,1e9)]:
    b=d[d['T']>lo].copy(); b['entry']=float(lo)
    b['E']=((b['T']<=hi)&(b['E']==1)).astype(int); b['T']=np.minimum(b['T'],hi)
    fm=CoxPHFitter(penalizer=1e-4).fit(b,'T','E',entry_col='entry')
    out[f'{lo}-{int(hi) if hi<1e8 else "+"} mo']=np.exp(fm.params_); print('band',lo,hi,'n',len(b),'events',int(b.E.sum()),flush=True)
H=pd.DataFrame(out); H['overall']=np.exp(f.params_); H.to_pickle('ph_bands.pkl'); print(H.round(2).to_string())
