import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
P=pickle.load(open('preds_all.pkl','rb'))
def km_risk(T,E,h):
    o=np.argsort(T,kind='stable'); T=T[o]; E=E[o]; n=len(T)
    if n==0: return np.nan
    atrisk=n-np.arange(n); s=1.0
    ut,idx=np.unique(T,return_index=True)
    ev=np.add.reduceat(E,idx); nr=n-idx
    m=ut<=h; return 1-np.prod(1-ev[m]/nr[m])
def nb_curve(T,E,risk,h,ths):
    Tc=np.minimum(T,h); Ec=((T<=h)&(E==1)).astype(float)
    out=[]; n=len(T); base=km_risk(Tc,Ec,h)
    for pt in ths:
        sel=risk>=pt; p=sel.mean()
        if sel.sum()<5: out.append(0.0); continue
        r=km_risk(Tc[sel],Ec[sel],h); out.append(r*p-(1-r)*p*pt/(1-pt))
    return np.array(out), base
ths=np.round(np.arange(0.02,0.601,0.01),3); names=['AJCC stage only','Cox, linear age','Cox, spline age','Gradient-boosted','DeepSurv','RSF']
res=[]
for which in ['SEER','TCGA']:
    d=(prep(te) if which=='SEER' else prep(tc.dropna(subset=['M_stage']))).reset_index(drop=True); T=d.survival_months.values; E=d.os_status.values
    for j,h in [(1,36),(2,60)]:
        base=None
        for k in names:
            risk=1-P[which][k][1][:,j]; nb,base=nb_curve(T,E,risk,h,ths)
            for pt,v in zip(ths,nb): res.append([which,h//12,k,pt,v])
        for pt in ths: res.append([which,h//12,'Treat all',pt,base-(1-base)*pt/(1-pt)]); res.append([which,h//12,'Treat none',pt,0.0])
R=pd.DataFrame(res,columns=['cohort','year','strategy','threshold','net_benefit']); R.to_csv('dca_all.csv',index=False)
pt=R.pivot_table(index=['cohort','year','threshold'],columns='strategy',values='net_benefit')
for (c,y) in [('SEER',5),('TCGA',5),('SEER',3)]:
    x=pt.loc[(c,y)].loc[[0.05,0.1,0.15,0.2,0.3,0.4,0.5]].round(4); print(c,y,'-year'); print(x.to_string())
# range where all models beat both references
for (c,y) in [('SEER',5),('TCGA',5),('SEER',3),('TCGA',3)]:
    x=pt.loc[(c,y)]; ok=(x[['Cox, linear age','Cox, spline age','Gradient-boosted','DeepSurv','RSF']].min(axis=1)>x[['Treat all','Treat none']].max(axis=1)); 
    print(c,y,'all five models > both references at',x.index[ok].min() if ok.any() else None,'to',x.index[ok].max() if ok.any() else None,'| n thresholds',ok.sum(),'of',len(x), '| treat-all<0 from',x.index[x['Treat all']<0].min())
