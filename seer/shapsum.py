import sys,pickle;sys.path.insert(0,'.')
import numpy as np,pandas as pd
from shapx import VARS
rows=[]
for f,tag in [('shapA.pkl','background resample (5 x 50 patients)'),('shapB.pkl','model refit')]:
    res=pickle.load(open(f,'rb')); keys=sorted({(k[1],k[2]) for k in res})
    for nm,coh in keys:
        A=np.array([v for k,v in res.items() if k[1]==nm and k[2]==coh]); rk=np.argsort(-A,axis=1)
        top=[tuple(r[:2]) for r in rk]; same=np.mean([r.tolist()==rk[0].tolist() for r in rk])
        for i,v in enumerate(VARS): rows.append([tag,nm,coh,len(A),v,round(A[:,i].mean(),3),round(A[:,i].min(),3),round(A[:,i].max(),3)])
        print(tag,nm,coh,'n=',len(A),'identical full ranking in',round(same,2),'of runs')
D=pd.DataFrame(rows,columns=['source','model','cohort','n_runs','variable','mean','min','max']); D.to_csv('shap_robustness.csv',index=False)
pd.set_option('display.width',200); print(D[D.variable.isin(['age','M_stage','T_stage','sex','N_stage'])].to_string(index=False))
