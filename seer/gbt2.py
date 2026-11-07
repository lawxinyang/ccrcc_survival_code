import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
from sklearn.tree import DecisionTreeRegressor
tr_=prep(tr); Xtr=X(tr_).values.astype(float); t=tr_.survival_months.values; e=tr_.os_status.values
b=np.random.default_rng(2000).integers(0,len(Xtr),len(Xtr)); Xb,tb,eb=Xtr[b],t[b],e[b]
cd=CoxData(tb,eb); F=np.zeros(len(b))
for m in range(216):
    g,h,_=cd.grad_hess(F); r=-g/h
    if not np.isfinite(r).all():
        print('round',m,'nonfinite r:',(~np.isfinite(r)).sum(),'F range',np.nanmin(F),np.nanmax(F),'g nan',np.isnan(g).sum(),'h nan',np.isnan(h).sum()); break
    F+=0.1*DecisionTreeRegressor(max_depth=3,random_state=42+m).fit(Xb,r,sample_weight=h).predict(Xb)
else: print('no failure')
print('max |r| in round', m, np.abs(r[np.isfinite(r)]).max())
