import sys;sys.path.insert(0,'.')
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from mlmodels import GBCox
from coxcore import CoxData
class GBCoxClip(GBCox):
    """Same as GBCox but each Newton step is capped at +/-clip (stabilises refits on bootstrap resamples, where the unclipped update diverged)."""
    def fit(s,X,t,e,clip=5.0):
        X=np.asarray(X,float); cd=CoxData(t,e); F=np.zeros(len(X)); s.trees=[]
        for m in range(s.n_rounds):
            g,h,_=cd.grad_hess(F); r=np.clip(-g/h,-clip,clip)
            tr=DecisionTreeRegressor(max_depth=s.depth,random_state=s.seed+m).fit(X,r,sample_weight=h)
            F+=s.lr*tr.predict(X); s.trees.append(tr)
        s.train_F=F; return s
