import pickle,glob,numpy as np,pandas as pd
lab=['C','AUC1y','AUC3y','AUC5y','Brier1y','Brier3y','Brier5y']
names=['AJCC stage only','Cox, linear age','Cox, spline age','Cox, spline + interactions','RSF','Gradient-boosted','DeepSurv']
out=[];pair=[]
for which in ['SEER','TCGA']:
    full=pickle.load(open(f'full2_{which}.pkl','rb')); bs=[]
    for f in sorted(glob.glob(f'boot2_{which}_*.pkl')): bs+=pickle.load(open(f,'rb'))
    arr={k:np.array([b[k] for b in bs]) for k in names}; print(which,'resamples',len(bs))
    for k in names:
        r={'cohort':which,'model':k}
        for l in ['C','AUC1y','AUC3y','AUC5y','Brier5y']:
            j=lab.index(l); lo,hi=np.percentile(arr[k][:,j],[2.5,97.5]); r[l]=f'{full[k][j]:.3f} ({lo:.3f}-{hi:.3f})'
        out.append(r)
    for a,b in [('Gradient-boosted','Cox, spline age'),('DeepSurv','Cox, spline age'),('RSF','Cox, spline age'),('Gradient-boosted','Cox, linear age'),('DeepSurv','Cox, linear age'),('Gradient-boosted','Cox, spline + interactions'),('DeepSurv','Cox, spline + interactions'),('Gradient-boosted','DeepSurv'),('Cox, spline age','AJCC stage only'),('Gradient-boosted','AJCC stage only')]:
        r={'cohort':which,'comparison':f'{a} minus {b}'}
        for l in ['C','AUC5y','Brier5y']:
            j=lab.index(l); d=arr[a][:,j]-arr[b][:,j]; lo,hi=np.percentile(d,[2.5,97.5]); p=min(1,2*min((d<=0).mean(),(d>=0).mean())); r[l]=f'{full[a][j]-full[b][j]:+.4f} ({lo:+.4f},{hi:+.4f})'; r[l+' P']=round(max(p,1/len(d)),3)
        pair.append(r)
O=pd.DataFrame(out); Pp=pd.DataFrame(pair); O.to_csv('final_table3_ci.csv',index=False); Pp.to_csv('final_pairwise.csv',index=False)
pd.set_option('display.width',280); print(O.to_string(index=False)); print(); print(Pp.to_string(index=False))
