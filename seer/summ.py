import pickle,numpy as np,pandas as pd,sys
which,setname=sys.argv[1],sys.argv[2]
full=pickle.load(open(f'full_{which}_{setname}.pkl','rb')); boots=pickle.load(open(f'boot_{which}_{setname}.pkl','rb'))
names=list(full); lab=['C','AUC1y','AUC3y','AUC5y','Brier1y','Brier3y','Brier5y']
arr={k:np.array([b[k] for b in boots]) for k in names}
print(which,'n boot',len(boots))
rows=[]
for k in names:
    r={'model':k}
    for i,l in enumerate(lab):
        lo,hi=np.percentile(arr[k][:,i],[2.5,97.5]); r[l]=f'{full[k][i]:.3f} ({lo:.3f}-{hi:.3f})' if l in('C','AUC5y','Brier5y') else f'{full[k][i]:.3f}'
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
ref='Cox, linear age (7 var)'
print('\nPaired differences vs',ref)
out=[]
for k in names:
    if k==ref: continue
    r={'model':k}
    for i,l in enumerate(['C','AUC5y','Brier5y']):
        j=lab.index(l); d=arr[k][:,j]-arr[ref][:,j]; lo,hi=np.percentile(d,[2.5,97.5]); p=2*min((d<=0).mean(),(d>=0).mean())
        r[l]=f'{full[k][j]-full[ref][j]:+.4f} ({lo:+.4f},{hi:+.4f}) p={max(p,1/len(d)):.3f}'
    out.append(r)
print(pd.DataFrame(out).to_string(index=False))
ref2='Cox, spline age (7 var)'
print('\nPaired differences vs',ref2)
out=[]
for k in names:
    if k==ref2: continue
    r={'model':k}
    for l in ['C','AUC5y','Brier5y']:
        j=lab.index(l); d=arr[k][:,j]-arr[ref2][:,j]; lo,hi=np.percentile(d,[2.5,97.5]); p=2*min((d<=0).mean(),(d>=0).mean())
        r[l]=f'{full[k][j]-full[ref2][j]:+.4f} ({lo:+.4f},{hi:+.4f}) p={max(p,1/len(d)):.3f}'
    out.append(r)
print(pd.DataFrame(out).to_string(index=False))
