import pickle,numpy as np,pandas as pd
lab=['C','AUC1y','AUC3y','AUC5y','Brier1y','Brier3y','Brier5y']
rows=[]; 
for which in ['SEER','TCGA']:
    f1=pickle.load(open(f'full_{which}_v1.pkl','rb')); f2=pickle.load(open(f'full_{which}_v2.pkl','rb'))
    b1=pickle.load(open(f'boot_{which}_v1.pkl','rb')); b2=pickle.load(open(f'boot_{which}_v2.pkl','rb'))
    B=min(len(b1),len(b2)); full={**f1,**f2}
    arr={k:np.array([b[k] for b in b1[:B]]) for k in f1}; arr.update({k:np.array([b[k] for b in b2[:B]]) for k in f2})
    ref='Cox, spline age (7 var)'; lin='Cox, linear age (7 var)'
    for k in full:
        r={'Cohort':which,'Model':k,'n boot':len(arr[k])}
        for l in ['C','AUC1y','AUC3y','AUC5y','Brier5y']:
            j=lab.index(l); lo,hi=np.percentile(arr[k][:,j],[2.5,97.5]); r[l]=round(full[k][j],4); r[l+' lo']=round(lo,4); r[l+' hi']=round(hi,4)
        for tag,rf in [('vs linear',lin),('vs spline',ref)]:
            for l in ['C','AUC5y','Brier5y']:
                j=lab.index(l); d=arr[k][:,j]-arr[rf][:,j]
                r[f'd{l} {tag}']=round(full[k][j]-full[rf][j],4); r[f'd{l} {tag} lo']=round(np.percentile(d,2.5),4); r[f'd{l} {tag} hi']=round(np.percentile(d,97.5),4)
        rows.append(r)
R=pd.DataFrame(rows); R.to_csv('cox_family_results.csv',index=False)
pd.set_option('display.width',250)
print(R[['Cohort','Model','C','C lo','C hi','dC vs spline','dC vs spline lo','dC vs spline hi']].to_string(index=False))
