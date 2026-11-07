import sys,time,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from mlmodels import *
import numpy as np
tr_=prep(tr); Xtr=X(tr_).values.astype(float); t=tr_.survival_months.values; e=tr_.os_status.values
b=np.random.default_rng(2000).integers(0,len(Xtr),len(Xtr))
cd=CoxData(t[b],e[b]); F=np.zeros(len(b))
g,h,_=cd.grad_hess(F); print('round0 finite',np.isfinite(-g/h).all(), 'min h',h.min(), 'zero-h share',(h<=1e-11).mean())
# where does NaN arise
t0=time.time()
try: GBCox().fit(Xtr[b],t[b],e[b])
except Exception as ex: print('fail',str(ex)[:60])
print(time.time()-t0)
