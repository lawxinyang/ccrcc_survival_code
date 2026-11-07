import sys,pickle,warnings,time;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from shapx import *
M=pickle.load(open('models_rebuilt.pkl','rb'));cols=M['cols'];rsf=pickle.load(open('rsf_model.pkl','rb'))
tr_,te_,tc_=prep(tr),prep(te),prep(tc.dropna(subset=['M_stage']))
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
Xtr,Xte,Xtc=mk(tr_),mk(te_),mk(tc_);idx=pickle.load(open('shap_out.pkl','rb'))['idx'][:300];grp=groups(cols)
bg=Xtr[np.random.default_rng(42).choice(len(Xtr),50,replace=False)]
Z=np.repeat(Xtr[:20000],1,axis=0);t0=time.time();rsf.predict(Z);print('20k rows predict sec',round(time.time()-t0,2),flush=True)
out={}
for coh,Xe in [('SEER',Xte[idx]),('TCGA',Xtc)]:
    phi,base,vf=exact_shap(rsf.predict,Xe,bg,grp);out[coh]=np.abs(phi).mean(0);print(coh,'additivity err',float(np.abs(phi.sum(1)+base-rsf.predict(Xe)).max()),'| sec',round(time.time()-t0),flush=True)
    print(dict(zip(VARS,np.round(out[coh],3))),flush=True)
pickle.dump(out,open('shapR.pkl','wb'))
