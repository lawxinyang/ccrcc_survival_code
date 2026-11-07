import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from lifelines.utils import concordance_index
out=pickle.load(open('norace_preds.pkl','rb'));rng=np.random.default_rng(11);rows=[]
for nm,d,B in [('SEER',prep(te),200),('TCGA',prep(tc.dropna(subset=['M_stage'])),1000)]:
    tt=d.survival_months.values;ee=d.os_status.values.astype(bool);n=len(d);idx=[rng.integers(0,n,n) for _ in range(B)]
    for mdl,(w,wo) in out[nm].items():
        C=lambda lp,i:concordance_index(tt[i],-lp[i],ee[i]);a=np.arange(n)
        dd=np.array([C(wo,i)-C(w,i) for i in idx]);rows.append([nm,mdl,round(C(w,a),4),round(C(wo,a),4),round(C(wo,a)-C(w,a),4),round(np.percentile(dd,2.5),4),round(np.percentile(dd,97.5),4)])
        print(rows[-1],flush=True)
R=pd.DataFrame(rows,columns=['cohort','model','C_with_race','C_without_race','diff','lo','hi']);R.to_csv('norace_ml.csv',index=False);print(R.to_string(index=False))
