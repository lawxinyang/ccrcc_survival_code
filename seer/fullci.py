import pickle,glob,numpy as np,pandas as pd
lab=['C-index','AUC 1y','AUC 3y','AUC 5y','Brier 1y','Brier 3y','Brier 5y']
rows=[]
for w in ['SEER','TCGA']:
    f=pickle.load(open(f'full2_{w}.pkl','rb'));bs=[]
    for p in sorted(glob.glob(f'boot2_{w}_*.pkl')): bs+=pickle.load(open(p,'rb'))
    for m in f:
        a=np.array([b[m] for b in bs]);r={'cohort':w,'model':m}
        for j,l in enumerate(lab): lo,hi=np.percentile(a[:,j],[2.5,97.5]);r[l]=f'{f[m][j]:.3f} ({lo:.3f}-{hi:.3f})'
        rows.append(r)
pd.DataFrame(rows).to_csv('Table3_all_horizons_CI.csv',index=False);print(pd.DataFrame(rows)[['cohort','model','AUC 1y','AUC 3y','Brier 1y','Brier 3y']].to_string(index=False))
