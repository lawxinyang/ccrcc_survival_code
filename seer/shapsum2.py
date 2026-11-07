import sys,pickle;sys.path.insert(0,'.')
import numpy as np,pandas as pd
from shapx import VARS
res=pickle.load(open('shapB.pkl','rb')); rows=[]
for nm in ['Gradient-boosted (80% refit)','DeepSurv (new seed)']:
    for coh in ['SEER','TCGA']:
        runs=[(k[0],v) for k,v in res.items() if k[1]==nm and k[2]==coh]
        ok=[v for r,v in runs if v.max()<5]; bad=[r for r,v in runs if v.max()>=5]
        A=np.array(ok); top3=[tuple(np.argsort(-a)[:3]) for a in A]
        print(nm,coh,'runs',len(runs),'diverged (excluded)',bad,'| top-3 (age,M,T) order kept in',sum(t==(0,6,4) for t in top3),'of',len(A))
        for i,v in enumerate(VARS): rows.append([nm,coh,len(A),v,round(A[:,i].mean(),3),round(A[:,i].min(),3),round(A[:,i].max(),3)])
D=pd.DataFrame(rows,columns=['model','cohort','n_valid_refits','variable','mean','min','max']); D.to_csv('shap_refit_uncertainty.csv',index=False)
print(D.to_string(index=False))
