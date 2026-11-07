import sys;sys.path.insert(0,'.')
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from coxcore import CoxData
class GBCox:
    def __init__(s,n_rounds=216,lr=0.1,depth=3,subsample=1.0,seed=42,min_child=1e-3):
        s.n_rounds,s.lr,s.depth,s.subsample,s.seed,s.min_child=n_rounds,lr,depth,subsample,seed,min_child
    def fit(s,X,t,e):
        X=np.asarray(X,float); cd=CoxData(t,e); F=np.zeros(len(X)); s.trees=[]; rng=np.random.default_rng(s.seed)
        for m in range(s.n_rounds):
            g,h,_=cd.grad_hess(F); r=-g/h
            idx=np.arange(len(X)) if s.subsample>=1 else rng.choice(len(X),int(s.subsample*len(X)),replace=False)
            tr=DecisionTreeRegressor(max_depth=s.depth,random_state=s.seed+m).fit(X[idx],r[idx],sample_weight=h[idx])
            F+=s.lr*tr.predict(X); s.trees.append(tr)
        s.train_F=F; return s
    def predict(s,X,n=None):
        X=np.asarray(X,float); trees=s.trees if n is None else s.trees[:n]
        return s.lr*sum(t.predict(X) for t in trees)
class DeepSurv:
    def __init__(s,hidden=8,lr=0.02,epochs=531,act='relu',seed=42):
        s.h,s.lr,s.ep,s.act,s.seed=hidden,lr,epochs,act,seed
    def _fwd(s,Xs):
        z=Xs@s.W1+s.b1; a=np.maximum(z,0) if s.act=='relu' else np.tanh(z); return z,a,(a@s.w2).ravel()+s.b2
    def fit(s,X,t,e):
        X=np.asarray(X,float); s.mu=X.mean(0); s.sd=X.std(0)+1e-8; Xs=(X-s.mu)/s.sd; n,p=Xs.shape; rng=np.random.default_rng(s.seed)
        s.W1=rng.normal(0,np.sqrt(2/p),(p,s.h)); s.b1=np.zeros(s.h); s.w2=rng.normal(0,np.sqrt(1/s.h),(s.h,1)); s.b2=0.0
        params=['W1','b1','w2']; m={k:0*getattr(s,k) for k in params}; v={k:0*getattr(s,k) for k in params}; cd=CoxData(t,e); s.hist=[]
        for ep in range(1,s.ep+1):
            z,a,f=s._fwd(Xs); g,_,ll=cd.grad_hess(f); gf=(g/n)[:,None]
            gw2=a.T@gf; ga=gf@s.w2.T; gz=ga*((z>0) if s.act=='relu' else (1-a**2)); gW1=Xs.T@gz; gb1=gz.sum(0)
            grads={'W1':gW1,'b1':gb1,'w2':gw2}
            for k in params:
                m[k]=0.9*m[k]+0.1*grads[k]; v[k]=0.999*v[k]+0.001*grads[k]**2
                setattr(s,k,getattr(s,k)-s.lr*(m[k]/(1-0.9**ep))/(np.sqrt(v[k]/(1-0.999**ep))+1e-8))
            s.hist.append(-ll/n)
        return s
    def predict(s,X): 
        Xs=(np.asarray(X,float)-s.mu)/s.sd; return s._fwd(Xs)[2]
