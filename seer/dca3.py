import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
exec(open('dca2.py').read().split("ths=np.round")[0])
THS=[0.10,0.20,0.30,0.40]; names=['AJCC stage only','Cox, spline age','Gradient-boosted','DeepSurv']
rows=[]
for which in ['SEER','TCGA']:
    d=(prep(te) if which=='SEER' else prep(tc.dropna(subset=['M_stage']))).reset_index(drop=True); T=d.survival_months.values; E=d.os_status.values
    for j,h in [(2,60),(1,36)]:
        risk={k:1-P[which][k][1][:,j] for k in names}
        full={k:nb_curve(T,E,risk[k],h,THS)[0] for k in names}
        rng=np.random.default_rng(11); B=400; bs={k:[] for k in names}
        for b in range(B):
            i=rng.integers(0,len(d),len(d)); 
            for k in names: bs[k].append(nb_curve(T[i],E[i],risk[k][i],h,THS)[0])
        bs={k:np.array(v) for k,v in bs.items()}
        for a,bname in [('Cox, spline age','AJCC stage only'),('Gradient-boosted','Cox, spline age'),('DeepSurv','Cox, spline age')]:
            for q,pt in enumerate(THS):
                dd=bs[a][:,q]-bs[bname][:,q]; rows.append([which,h//12,pt,f'{a} minus {bname}',round(full[a][q]-full[bname][q],4),round(np.percentile(dd,2.5),4),round(np.percentile(dd,97.5),4)])
R=pd.DataFrame(rows,columns=['cohort','year','threshold','difference','NB_diff','lo','hi']); R.to_csv('dca_diff_ci.csv',index=False)
print(R[R.year==5].to_string(index=False))
