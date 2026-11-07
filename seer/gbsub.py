import sys;sys.path.insert(0,'.')
import warnings;warnings.filterwarnings('ignore')
from common import *; from mlmodels import *
tr_=prep(tr); Xtr=X(tr_).values.astype(float); t=tr_.survival_months.values; e=tr_.os_status.values
for s in range(4):
    i=np.random.default_rng(3000+s).choice(len(Xtr),int(.8*len(Xtr)),replace=False)
    try:
        g=GBCox().fit(Xtr[i],t[i],e[i]); print(s,'ok',np.abs(g.predict(Xtr[:2000])).max(),flush=True)
    except Exception as ex: print(s,'fail',flush=True)
