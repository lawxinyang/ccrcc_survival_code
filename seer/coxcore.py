import numpy as np
class CoxData:
    """Breslow partial likelihood pieces for data sorted by time."""
    def __init__(s,t,e):
        o=np.argsort(t,kind='stable'); s.o=o; s.t=np.asarray(t,float)[o]; s.e=np.asarray(e,float)[o]; n=len(s.t); s.n=n
        # group ids of unique times
        _,s.first,s.inv=np.unique(s.t,return_index=True,return_inverse=True)
        s.ng=len(s.first)
    def grad_hess(s,f):
        f=np.asarray(f,float)[s.o]; f=f-f.max(); ef=np.exp(f)
        rev=np.cumsum(ef[::-1])[::-1]                 # sum_{j>=i}
        S=rev[s.first][s.inv]                         # risk-set sum at tie-group start
        # per-group: events count and 1/S
        d_g=np.bincount(s.inv,weights=s.e,minlength=s.ng)
        Sg=rev[s.first]
        a=np.cumsum(d_g/Sg); b=np.cumsum(d_g/Sg**2)
        A=a[s.inv]; B=b[s.inv]
        g=-s.e+ef*A                                   # d(-loglik)/df
        h=ef*A-ef**2*B
        out_g=np.empty(s.n); out_h=np.empty(s.n); out_g[s.o]=g; out_h[s.o]=h
        ll=np.sum(s.e*(f-np.log(S)))
        return out_g,np.maximum(out_h,1e-12),ll
