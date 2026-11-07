import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from lifelines import CoxPHFitter
tr_=prep(tr); keep=(tr_['T_stage']!='T0').values
M=X(tr_).drop(columns=['T_stage_T0'])[keep].reset_index(drop=True)
T=tr_.survival_months.values[keep]; E=tr_.os_status.values[keep]
f=CoxPHFitter().fit(M.assign(T=T,E=E),'T','E'); out={}; info=[]
for lo,hi in [(0,12),(12,36),(36,60),(60,1e9)]:
    a=T>lo; b=M[a].copy(); t=T[a]-lo; e=E[a]; w=hi-lo
    e2=((t<=w)&(e==1)).astype(int); t2=np.minimum(t,w)
    fm=CoxPHFitter(penalizer=1e-4).fit(b.assign(T=t2,E=e2),'T','E')
    out[f'{lo}-{int(hi) if hi<1e8 else "+"} mo']=np.exp(fm.params_); info.append((lo,hi,int(a.sum()),int(e2.sum())))
H=pd.DataFrame(out); H['overall']=np.exp(f.params_); H.to_pickle('ph_bands.pkl'); print(info); print(H.round(2).to_string())
