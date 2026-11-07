import numpy as np, itertools, math
VARS=['age','sex','race','grade','T_stage','N_stage','M_stage']
def groups(cols):
    g={v:[i for i,c in enumerate(cols) if c==v or c.startswith(v+'_')] for v in VARS}
    # age_rcs columns (spline model) belong to age
    for i,c in enumerate(cols):
        if c.startswith('age*'): pass
    return g
def exact_shap(f,Xexp,Xbg,grp,chunk=50):
    """Exact interventional Shapley values over the 7 original variables (all encoded columns of a variable are masked jointly).
    f: callable (n x p array) -> log-risk. Returns (N x 7) phi and base value E[f(bg)]."""
    n=len(VARS); masks=list(range(2**n)); B=len(Xbg)
    w={}  # Shapley weights by coalition size
    for s in range(n): w[s]=math.factorial(s)*math.factorial(n-s-1)/math.factorial(n)
    colsets=[ [c for i,v in enumerate(VARS) if (m>>i)&1 for c in grp[v]] for m in masks]
    phi=np.zeros((len(Xexp),n)); vfull=np.zeros(len(Xexp)); base=None
    for s0 in range(0,len(Xexp),chunk):
        xe=Xexp[s0:s0+chunk]; N=len(xe)
        V=np.zeros((N,len(masks)))
        big=np.repeat(Xbg[None,:,:],N*len(masks),axis=0).reshape(len(masks),N,B,-1) if False else None
        for mi,cs in enumerate(colsets):
            Z=np.broadcast_to(Xbg[None,:,:],(N,B,Xbg.shape[1])).copy()
            if cs: Z[:,:,cs]=xe[:,None,cs]
            V[:,mi]=f(Z.reshape(N*B,-1)).reshape(N,B).mean(1)
        for i in range(n):
            acc=np.zeros(N)
            for mi in masks:
                if (mi>>i)&1: continue
                s=bin(mi).count('1'); acc+=w[s]*(V[:,mi|(1<<i)]-V[:,mi])
            phi[s0:s0+N,i]=acc
        vfull[s0:s0+N]=V[:,-1]; base=V[0,0]
    return phi,base,vfull
