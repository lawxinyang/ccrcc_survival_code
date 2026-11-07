import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from variants import *
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
M=pickle.load(open('models_rebuilt.pkl','rb')); cols=M['cols']; rsf=pickle.load(open('rsf_model.pkl','rb'))
tr_=prep(tr);te_=prep(te);tc_=prep(tc.dropna(subset=['M_stage']))
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
cox=fit_all({'Cox':fx_spl})['Cox']; t=tr_.survival_months.values;e=tr_.os_status.values
def breslow(lp,t,e):
    o=np.argsort(t);tt=t[o];ee=e[o];r=np.exp(lp[o]);S=np.cumsum(r[::-1])[::-1]
    def H(x):
        m=(tt<=x)&(ee==1);return np.sum(1/S[m]) if False else None
    # grouped by unique event times
    ut=np.unique(tt[ee==1]);d=np.array([ee[tt==u].sum() for u in ut]);Sr=np.array([r[tt>=u].sum() for u in ut]);cum=np.cumsum(d/Sr)
    return lambda x:cum[np.searchsorted(ut,x,side='right')-1] if x>=ut[0] else 0.0
def risk5(name,d):
    if name=='RSF':
        Xd=mk(d);out=[]
        for s in range(0,len(Xd),5000):
            fns=rsf.predict_survival_function(Xd[s:s+5000]);out+= [1-f(60.0) for f in fns]
        return np.array(out)
    if name=='Cox': fx,c,m=cox;lp=m.predict(fx(d).reindex(columns=c,fill_value=0.0));lptr=m.predict(fx(tr_).reindex(columns=c,fill_value=0.0))
    else: f=M['gb' if name=='GB' else 'ds'].predict;lp=f(mk(d));lptr=f(mk(tr_))
    H=breslow(lptr,t,e)(60.0);return 1-np.exp(-np.exp(lp)*H)
names=['Cox','RSF','GB','DS'];LAB={'Cox':'Cox (spline age)','RSF':'Random survival forest','GB':'Gradient-boosted Cox','DS':'DeepSurv'}
R={nm:{k:risk5(nm,d) for k,d in [('train',tr_),('SEER',te_),('TCGA',tc_)]} for nm in names}
# consistency check with stored 5-year predictions on SEER test
P=pickle.load(open('preds_all.pkl','rb'))['SEER']
for nm,key in [('Cox','Cox, spline age'),('GB','Gradient-boosted'),('DS','DeepSurv'),('RSF','RSF')]:
    print(nm,'max |diff| vs stored 5-y risk on SEER test',round(float(np.abs(R[nm]['SEER']-(1-P[key][1][:,2])).max()),4))
cut={};rows=[]
for nm in names:
    c=np.quantile(R[nm]['train'],[1/3,2/3]);cut[nm]=c
    for coh,d in [('SEER',te_),('TCGA',tc_)]:
        g=np.digitize(R[nm][coh],c);res=multivariate_logrank_test(d.survival_months,g,d.os_status)
        for k,l in enumerate(['Low','Intermediate','High']):
            m=g==k;km=KaplanMeierFitter().fit(d.survival_months[m],d.os_status[m])
            s5=km.survival_function_at_times(60).iloc[0];rows.append([LAB[nm],coh,l,int(m.sum()),round(c[0],3),round(c[1],3),round(100*(1-s5),1),round(res.test_statistic,1),res.p_value])
T=pd.DataFrame(rows,columns=['model','cohort','group','n','cut1_5y_risk','cut2_5y_risk','observed_5y_risk_%_KM','logrank_chi2_2df','P']);print(T.to_string(index=False));T.to_csv('R111_risk_tertiles.csv',index=False)
pickle.dump({'R':R,'cut':cut},open('tert.pkl','wb'))
col=['#2b6cb0','#d69e2e','#c53030']
for coh,d,tmax,step in [('SEER',te_,120,12),('TCGA',tc_,120,24)]:
    fig=plt.figure(figsize=(11,11)); gs=fig.add_gridspec(4,2,height_ratios=[3,0.9,3,0.9],hspace=.55)
    plt.rcParams['font.family']='Liberation Serif'
    for i,nm in enumerate(names):
        r,cc=divmod(i,2);ax=fig.add_subplot(gs[2*r,cc]);ax2=fig.add_subplot(gs[2*r+1,cc])
        g=np.digitize(R[nm][coh],cut[nm]);ticks=np.arange(0,tmax+1,step)
        for k,l in enumerate(['Low','Intermediate','High']):
            m=g==k;km=KaplanMeierFitter().fit(d.survival_months[m],d.os_status[m],label=l);km.plot_survival_function(ax=ax,ci_show=True,color=col[k],ci_alpha=.2,show_censors=False,lw=1.6)
            for x in ticks: ax2.text(x,2-k,str(int((d.survival_months[m]>=x).sum())),ha='center',va='center',fontsize=8,color=col[k])
        chi=T[(T.model==LAB[nm])&(T.cohort==coh)].logrank_chi2_2df.iloc[0]
        ax.set_xlim(-8,tmax+4);ax.set_ylim(0,1.02);ax.set_xticks(ticks);ax.set_xlabel('');ax.set_ylabel('Overall survival probability');ax.set_title(f"{'ABCD'[i]}. {LAB[nm]}  (log-rank χ² = {chi:,.0f}, P < .0001)",loc='left',fontsize=10);ax.legend(frameon=False,fontsize=8,loc='lower left');ax.spines[['top','right']].set_visible(False)
        ax2.set_xlim(-8,tmax+4);ax2.set_ylim(-.6,2.6);ax2.set_xticks(ticks);ax2.set_yticks([2,1,0]);ax2.set_yticklabels(['Low','Int.','High'],fontsize=8);ax2.set_xlabel('Months since diagnosis');ax2.set_title('Number at risk',loc='left',fontsize=8);ax2.spines[['top','right']].set_visible(False)
    fig.savefig(f'Figure_KM_{coh}_rebuilt.png',dpi=300,bbox_inches='tight');fig.savefig(f'Figure_KM_{coh}_rebuilt.pdf',bbox_inches='tight')
