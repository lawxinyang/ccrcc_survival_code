import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from variants import *
V={'AJCC stage only':fx_stage,'Cox, linear age':fx_lin,'Cox, spline age':fx_spl,'Cox, spline + interactions':fx_ext}
models=fit_all(V); old=pickle.load(open('preds_rebuilt.pkl','rb'))
out={}
for nm,d in [('SEER',prep(te)),('TCGA',prep(tc.dropna(subset=['M_stage'])))]:
    P=predict(models,d); P2={k:v for k,v in P.items()}
    for k in ['Gradient-boosted','DeepSurv','RSF']: P2[k]=old[nm][k]
    P2.pop('Cox, linear age',None); P2['Cox, linear age']=P['Cox, linear age']
    out[nm]=P2
pickle.dump(out,open('preds_all.pkl','wb')); print({n:list(v) for n,v in out.items()})
