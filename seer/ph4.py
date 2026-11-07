import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
tr_=prep(tr); keep=(tr_['T_stage']!='T0').values
M=X(tr_).drop(columns=['T_stage_T0'])[keep].reset_index(drop=True)
T=tr_.survival_months.values[keep]; E=tr_.os_status.values[keep]
m=CoxPHSurvivalAnalysis(alpha=1e-5,ties='breslow',n_iter=200).fit(M,Surv.from_arrays(E.astype(bool),T)); out={}
for lo,hi in [(0,12),(12,36),(36,60),(60,1e9)]:
    a=T>lo; t=T[a]-lo; e=E[a]; w=hi-lo
    e2=((t<=w)&(e==1)); t2=np.minimum(t,w)
    fm=CoxPHSurvivalAnalysis(alpha=1e-5,ties='breslow',n_iter=200).fit(M[a],Surv.from_arrays(e2,t2))
    out[f'{lo}-{int(hi) if hi<1e8 else "+"} mo']=pd.Series(np.exp(fm.coef_),index=M.columns)
H=pd.DataFrame(out); H['overall']=np.exp(m.coef_); H.to_pickle('ph_bands.pkl'); print(H.round(2).to_string())
