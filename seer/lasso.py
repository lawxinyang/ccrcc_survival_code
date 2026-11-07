import sys,pickle,warnings,time;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
tr_=prep(tr);V8=('age','sex','race','laterality','grade','T_stage','N_stage','M_stage')
REF2=dict(REF);REF2['laterality']='Right'
Xd=X(tr_,V8,REF2);cols=list(Xd.columns);print(len(cols),cols)
y=Surv.from_arrays(tr_.os_status.astype(bool),tr_.survival_months)
sc=StandardScaler().fit(Xd.values);Z=sc.transform(Xd.values)
# alpha grid: glmnet-style, lambda_max down to 1e-4 * lambda_max, 100 values
probe=CoxnetSurvivalAnalysis(l1_ratio=1.0,n_alphas=150,alpha_min_ratio=1e-7,max_iter=100000,fit_baseline_model=False).fit(Z,y)
alphas=np.geomspace(probe.alphas_[0],5e-4,50);print('lambda_max',alphas[0],'lambda_min_grid',alphas[-1])
def harrell(lp,t,e):
    from sksurv.metrics import concordance_index_censored
    return concordance_index_censored(e.astype(bool),t,lp)[0]
skf=StratifiedKFold(5,shuffle=True,random_state=42);cv=np.zeros((5,len(alphas)));t=tr_.survival_months.values;e=tr_.os_status.values
for f,(a,b) in enumerate(skf.split(Z,e)):
    for j,al in enumerate(alphas):
        m=CoxnetSurvivalAnalysis(l1_ratio=1.0,alphas=[al],max_iter=100000,fit_baseline_model=False).fit(Z[a],y[a]);cv[f,j]=harrell(Z[b]@m.coef_[:,0],t[b],e[b])
    print('fold',f,'done',flush=True)
mu=cv.mean(0);se=cv.std(0,ddof=1)/np.sqrt(5);jmin=int(np.argmax(mu));j1=int(np.where(mu>=mu[jmin]-se[jmin])[0][0])
print('lambda.min',alphas[jmin],'CV C',round(mu[jmin],4),'SE',round(se[jmin],5),'| lambda.1se',alphas[j1],'CV C',round(mu[j1],4))
B=np.column_stack([CoxnetSurvivalAnalysis(l1_ratio=1.0,alphas=[al],max_iter=100000,fit_baseline_model=False).fit(Z,y).coef_[:,0] for al in alphas])  # standardized coefs, p x nalpha
def var(c): return next(v for v in V8 if c==v or c.startswith(v+'_'))
rows=[]
for nm,j in [('lambda.min',jmin),('lambda.1se',j1)]:
    for c,b in zip(cols,B[:,j]): rows.append([nm,alphas[j],var(c),c,b,b/sc.scale_[cols.index(c)]])
R=pd.DataFrame(rows,columns=['rule','lambda','variable','term','coef_standardized','coef_original_scale']);R.to_csv('lasso_coefficients_rebuilt.csv',index=False)
for nm in ['lambda.min','lambda.1se']:
    d=R[R.rule==nm];kept=sorted(set(d[d.coef_standardized!=0].variable),key=V8.index);zero=[v for v in V8 if v not in kept]
    print(nm,'variables retained',len(kept),kept,'| dropped',zero,'| nonzero terms',int((d.coef_standardized!=0).sum()),'of',len(d))
# first entry lambda for each variable (path order)
entry={}
for v in V8:
    idx=[i for i,c in enumerate(cols) if var(c)==v];nz=np.where((B[idx,:]!=0).any(0))[0];entry[v]=alphas[nz[0]] if len(nz) else None
print('entry lambda by variable',{k:(round(x,5) if x else None) for k,x in entry.items()})
pickle.dump({'alphas':alphas,'cv':cv,'mu':mu,'se':se,'B':B,'cols':cols,'jmin':jmin,'j1':j1},open('lasso_out.pkl','wb'))
# figures
plt.rcParams.update({'font.family':'Liberation Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
LAB={'age':'Age','sex':'Sex','race':'Race','laterality':'Laterality','grade':'Grade','T_stage':'T stage','N_stage':'N stage','M_stage':'M stage'}
colr=dict(zip(V8,['#000000','#e69f00','#56b4e9','#d62728','#009e73','#0072b2','#cc79a7','#7f7f7f']))
fig,ax=plt.subplots(1,2,figsize=(11,4.2));x=np.log10(alphas)
for i,c in enumerate(cols): ax[0].plot(x,B[i],color=colr[var(c)],lw=1.2)
for v in V8: ax[0].plot([],[],color=colr[v],label=LAB[v])
ax[0].axvline(np.log10(alphas[jmin]),ls='--',color='k',lw=.8);ax[0].axvline(np.log10(alphas[j1]),ls=':',color='k',lw=.8)
ax[0].set_xlabel('log$_{10}$(λ)');ax[0].set_ylabel('Coefficient (per 1 SD of the standardized term)');ax[0].set_title('A. Coefficient paths (one line per model term)',loc='left');ax[0].legend(frameon=False,fontsize=8,ncol=2);ax[0].invert_xaxis()
ax[1].errorbar(x,mu,yerr=se,fmt='o',ms=3,color='#4a4a4a',ecolor='#9a9a9a',capsize=0);ax[1].axvline(np.log10(alphas[jmin]),ls='--',color='k',lw=.8,label=f'λ$_{{min}}$ = {alphas[jmin]:.5f}');ax[1].axvline(np.log10(alphas[j1]),ls=':',color='k',lw=.8,label=f'λ$_{{1SE}}$ = {alphas[j1]:.5f}')
ax[1].set_xlabel('log$_{10}$(λ)');ax[1].set_ylabel('Cross-validated C-index (mean ± SE, 5-fold)');ax[1].set_title('B. Cross-validated C-index',loc='left');ax[1].legend(frameon=False,fontsize=8);ax[1].invert_xaxis();ax[1].set_ylim(mu.min()-.002 if mu.min()>.7 else .70,mu.max()+.004)
plt.tight_layout();plt.savefig('Figure_LASSO_rebuilt.png',dpi=300);plt.savefig('Figure_LASSO_rebuilt.pdf')
