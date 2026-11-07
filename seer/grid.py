import sys,time,warnings,pickle,itertools,json;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import GBCox,DeepSurv
from gbclip import GBCoxClip
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
which=sys.argv[1]
tr_=prep(tr);Xa=X(tr_).values;t=tr_.survival_months.values;e=tr_.os_status.values.astype(int)
kf=KFold(5,shuffle=True,random_state=42);folds=list(kf.split(Xa))
grids={'RSF':[(d,l) for d in (3,5,7) for l in (20,50,100)],
       'GB':[(lr,d) for lr in (0.05,0.1,0.2) for d in (2,3,4)],
       'GBC':[(lr,d) for lr in (0.05,0.1,0.2) for d in (2,3,4)],
       'DS':[(h,lr) for h in (4,8,16) for lr in (0.01,0.02,0.05)]}
res=[]
for p in grids[which]:
    cs=[];t0=time.time()
    for tri,vai in folds:
        try:
            if which=='RSF':
                m=RandomSurvivalForest(n_estimators=105,max_depth=p[0],min_samples_leaf=p[1],max_features='sqrt',n_jobs=1,random_state=42).fit(Xa[tri],Surv.from_arrays(e[tri].astype(bool),t[tri])); s=m.predict(Xa[vai])
            elif which=='GB':
                m=GBCox(216,p[0],p[1]).fit(Xa[tri],t[tri],e[tri]); s=m.predict(Xa[vai])
            elif which=='GBC':
                m=GBCoxClip(216,p[0],p[1]).fit(Xa[tri],t[tri],e[tri],clip=50.0); s=m.predict(Xa[vai])
            else:
                m=DeepSurv(p[0],p[1],531).fit(Xa[tri],t[tri],e[tri]); s=m.predict(Xa[vai])
            c=concordance_index(t[vai],-s,e[vai]) if np.isfinite(s).all() else np.nan
        except Exception as ex: c=np.nan
        cs.append(c)
    res.append((p,float(np.nanmean(cs)) if not np.isnan(cs).all() else np.nan,int(np.isnan(cs).sum())))
    print(which,p,'CV C',round(res[-1][1],4),'failed folds',res[-1][2],'sec',round(time.time()-t0),flush=True)
pickle.dump(res,open(f'grid_{which}.pkl','wb'))
