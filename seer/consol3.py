import pickle,glob,numpy as np,pandas as pd
lab=['C-index','AUC 1y','AUC 3y','AUC 5y','Brier 1y','Brier 3y','Brier 5y'];R='RSF (revised scoring)'
rows=[];pair=[]
for w in ['SEER','TCGA']:
    f2=pickle.load(open(f'full2_{w}.pkl','rb'));f3=pickle.load(open(f'full3_{w}.pkl','rb'))
    b2=[];b3=[]
    for p in sorted(glob.glob(f'boot2_{w}_*.pkl')): b2+=pickle.load(open(p,'rb'))
    for p in sorted(glob.glob(f'boot3_{w}_*.pkl')): b3+=pickle.load(open(p,'rb'))
    assert len(b2)==len(b3),(len(b2),len(b3))
    arr={k:np.array([b[k] for b in b2]) for k in f2};arr[R]=np.array([b[R] for b in b3]);full={**f2,**f3}
    r={'cohort':w,'model':R}
    for j,l in enumerate(lab): lo,hi=np.percentile(arr[R][:,j],[2.5,97.5]);r[l]=f'{full[R][j]:.3f} ({lo:.3f}-{hi:.3f})'
    rows.append(r)
    for b in ['Cox, spline age','Cox, linear age','Gradient-boosted','DeepSurv','AJCC stage only']:
        q={'cohort':w,'comparison':f'{R} minus {b}'}
        for l in ['C-index','AUC 5y','Brier 5y']:
            j=lab.index(l);d=arr[R][:,j]-arr[b][:,j];lo,hi=np.percentile(d,[2.5,97.5]);p=min(1,2*min((d<=0).mean(),(d>=0).mean()))
            q[l]=f'{full[R][j]-full[b][j]:+.4f} ({lo:+.4f},{hi:+.4f})';q[l+' P']=round(max(p,1/len(d)),3)
        pair.append(q)
pd.DataFrame(rows).to_csv('rsf_revised_ci.csv',index=False);pd.DataFrame(pair).to_csv('rsf_revised_pairwise.csv',index=False)
pd.set_option('display.width',260);print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(pair).to_string(index=False))
