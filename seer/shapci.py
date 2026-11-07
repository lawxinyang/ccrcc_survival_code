import sys,pickle;sys.path.insert(0,'.')
import numpy as np,pandas as pd
from shapx import VARS
o=pickle.load(open('shap_out.pkl','rb'))['out']; rng=np.random.default_rng(5); rows=[]; ranks=[]
for (nm,coh),(phi,_,_) in o.items():
    a=np.abs(phi); n=len(a); B=1000; bs=np.array([a[rng.integers(0,n,n)].mean(0) for _ in range(B)])
    est=a.mean(0); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    for i,v in enumerate(VARS): rows.append([nm,coh,v,round(est[i],3),round(lo[i],3),round(hi[i],3),int((est>est[i]).sum())+1])
    order=np.argsort(-bs,axis=1); ref=tuple(np.argsort(-est)[:4])
    ranks.append((nm,coh,round(np.mean([tuple(r[:4])==ref for r in order]),3)))
R=pd.DataFrame(rows,columns=['model','cohort','variable','mean_abs_SHAP','lo','hi','rank']); R.to_csv('shap_importance_ci.csv',index=False)
print(R[R.model=='Gradient-boosted'].sort_values(['cohort','rank']).to_string(index=False)); print(ranks)
