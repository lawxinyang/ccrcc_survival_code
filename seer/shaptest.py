import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from metrics import fit_cox
from shapx import *
tr_=prep(tr); Xt=X(tr_); cols=list(Xt.columns); m=fit_cox(Xt,tr_)
grp=groups(cols); print({k:len(v) for k,v in grp.items()}, 'cols assigned',sum(len(v) for v in grp.values()),'of',len(cols))
rng=np.random.default_rng(1); bg=Xt.values[rng.choice(len(Xt),50,replace=False)]
te_=prep(te); Xe=X(te_).reindex(columns=cols,fill_value=0.0).values[:40]
f=lambda Z:Z@m.coef_
phi,base,vfull=exact_shap(f,Xe,bg,grp)
# analytic: linear model => phi_v = sum_{cols in v} beta*(x - mean(bg))
ana=np.column_stack([ (Xe[:,grp[v]]-bg[:,grp[v]].mean(0))@m.coef_[grp[v]] for v in VARS])
print('max |exact - analytic|',np.abs(phi-ana).max(),' additivity err',np.abs(phi.sum(1)+base-f(Xe)).max())
