import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
P=pickle.load(open('preds_all.pkl','rb'))['TCGA'];d=prep(tc.dropna(subset=['M_stage']));y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months)
lp,S=P['RSF']
print('RSF lp-based AUC (stored lp):',np.round(cumulative_dynamic_auc(y,y,lp,TIMES)[0],3))
for j,h in enumerate(TIMES): print(h,'AUC using 1-S(h) as score:',round(cumulative_dynamic_auc(y,y,1-S[:,j],np.array([h]))[0][0],3))
for nm in ['Cox, linear age','Gradient-boosted','DeepSurv']:
    lp,S=P[nm];print(nm,[round(cumulative_dynamic_auc(y,y,1-S[:,j],np.array([h]))[0][0],3) for j,h in enumerate(TIMES)])
