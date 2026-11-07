import sys,pickle,warnings,time;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
from gbclip import GBCoxClip
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
M=pickle.load(open('models_rebuilt.pkl','rb'));cols=M['cols'];rsf=pickle.load(open('rsf_model.pkl','rb'))
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
keep=[i for i,c in enumerate(cols) if not c.startswith('race')];print('terms',len(cols),'->',len(keep))
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_);t=tr_.survival_months.values;e=tr_.os_status.values;t0=time.time()
gbf=GBCoxClip().fit(Xtr,t,e,clip=50);gb=GBCoxClip().fit(Xtr[:,keep],t,e,clip=50);print('GB',round(time.time()-t0),flush=True)
ds=DeepSurv(seed=42).fit(Xtr[:,keep],t,e);print('DS',round(time.time()-t0),flush=True)
rf=RandomSurvivalForest(n_estimators=105,max_depth=5,min_samples_leaf=50,max_features='sqrt',n_jobs=1,random_state=42).fit(Xtr[:,keep],Surv.from_arrays(e.astype(bool),t));print('RSF',round(time.time()-t0),flush=True)
out={}
for nm,Xe in [('SEER',Xte),('TCGA',Xtc)]:
    out[nm]={'GB (step cap 50 in both fits)':(gbf.predict(Xe),gb.predict(Xe[:,keep])),'DS':(M['ds'].predict(Xe),ds.predict(Xe[:,keep])),'RSF':(rsf.predict(Xe),rf.predict(Xe[:,keep]))}
pickle.dump(out,open('norace_preds.pkl','wb'))
rng=np.random.default_rng(11);rows=[]
for nm,d,B in [('SEER',te_,300),('TCGA',tc_,1000)]:
    tt=d.survival_months.values;ee=d.os_status.values.astype(bool);n=len(d);idx=[rng.integers(0,n,n) for _ in range(B)]
    for mdl,(w,wo) in out[nm].items():
        C=lambda lp,i:concordance_index_censored(ee[i],tt[i],lp[i])[0]
        full=C(w,np.arange(n));nr=C(wo,np.arange(n));dd=np.array([C(wo,i)-C(w,i) for i in idx])
        rows.append([nm,mdl,round(full,4),round(nr,4),round(nr-full,4),round(np.percentile(dd,2.5),4),round(np.percentile(dd,97.5),4)])
R=pd.DataFrame(rows,columns=['cohort','model','C_with_race','C_without_race','diff','lo','hi']);print(R.to_string(index=False));R.to_csv('norace_ml.csv',index=False)
