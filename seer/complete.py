import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from lifelines.utils import concordance_index
P=pickle.load(open('preds_all.pkl','rb')); rsf=pickle.load(open('rsf_model.pkl','rb'))
tr_=prep(tr);cols=list(X(tr_).columns);mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
rows=[]
for coh,d in [('SEER',prep(te)),('TCGA',prep(tc.dropna(subset=['M_stage'])))]:
    s={k:P[coh][k][0] for k in ['AJCC stage only','Cox, spline age','Gradient-boosted','DeepSurv']}; s['RSF']=rsf.predict(mk(d))
    keep=((d.grade!='GX')&(~d.T_stage.isin(['T0','TX']))&(d.N_stage!='NX')&(d.M_stage!='MX')).values
    t=d.survival_months.values[keep];e=d.os_status.values[keep];n=keep.sum()
    S={k:v[keep] for k,v in s.items()}
    C=lambda k,ix:concordance_index(t[ix],-S[k][ix],e[ix])
    full=np.arange(n);B=300;rng=np.random.default_rng(42);bs={k:[] for k in S}
    for b in range(B):
        ix=rng.integers(0,n,n)
        for k in S: bs[k].append(C(k,ix))
    bs={k:np.array(v) for k,v in bs.items()}
    for k in S:
        c=C(k,full);lo,hi=np.percentile(bs[k],[2.5,97.5]);rows.append([coh,n,int(e.sum()),k,round(c,4),round(lo,4),round(hi,4),'',''])
        if k in('Gradient-boosted','DeepSurv','RSF'):
            dd=bs[k]-bs['Cox, spline age'];rows[-1][7]=round(C(k,full)-C('Cox, spline age',full),4);rows[-1][8]=f"{np.percentile(dd,2.5):.4f} to {np.percentile(dd,97.5):.4f}"
    print(coh,n,int(e.sum()),flush=True)
import pandas as pd
R=pd.DataFrame(rows,columns=['cohort','n','deaths','model','C-index','lo','hi','diff vs spline Cox','95% CI of diff']);R.to_csv('/mnt/user-data/outputs/Sensitivity_complete_coded_records.csv',index=False);print(R.to_string(index=False))
