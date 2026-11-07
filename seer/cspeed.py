import time,numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
rng=np.random.default_rng(0);n=26000;t=rng.exponential(50,n);e=rng.random(n)<.45;lp=-t+rng.normal(0,30,n)
t0=time.time();a=concordance_index_censored(e,t,lp)[0];t1=time.time();b=concordance_index(t,-lp,e);t2=time.time();print(round(t1-t0,2),round(t2-t1,2),a,b)
