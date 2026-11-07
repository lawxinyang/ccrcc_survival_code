import sys;sys.path.insert(0,'.')
from common import *
from lifelines import CoxPHFitter
tr_=prep(tr)
res={}
for name,q in [('3 knots (10/50/90)',[.10,.5,.90]),('4 knots (5/35/65/95)',[.05,.35,.65,.95]),('4 knots (10/33/67/90)',[.10,1/3,2/3,.90]),('5 knots (5/27.5/50/72.5/95)',[.05,.275,.5,.725,.95])]:
    kn=np.quantile(tr_.age,q)
    cols=X(tr_).columns
    Xs=X_spline(tr_,kn); Xl=X(tr_)
    out={}
    for tag,M in [('lin',Xl),('spl',Xs)]:
        d=M.copy(); d['T']=tr_.survival_months.values; d['E']=tr_.os_status.values
        f=CoxPHFitter().fit(d,'T','E'); out[tag]=f.log_likelihood_
    lr=2*(out['spl']-out['lin']); df=Xs.shape[1]-Xl.shape[1]+0
    print(name,'knots',np.round(kn,1),'extra df',df,'LR chi2',round(lr,1))
