import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from metrics import *
from lifelines.utils import concordance_index
rsf=pickle.load(open('rsf_model.pkl','rb'));M=pickle.load(open('models_rebuilt.pkl','rb'));cols=M['cols']
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values
P=pickle.load(open('preds_all.pkl','rb'))
trp=prep(tr);ytr=Surv.from_arrays(trp.os_status.astype(bool),trp.survival_months)
for coh,d,ref in [('SEER',prep(te),ytr),('TCGA',prep(tc.dropna(subset=['M_stage'])),None)]:
    y=Surv.from_arrays(d.os_status.astype(bool),d.survival_months);r=ytr if ref is not None else y
    lp,S=P[coh]['RSF'];Xd=mk(d);native=rsf.predict(Xd)
    fns=rsf.predict_survival_function(Xd);grid=rsf.unique_times_;last=np.array([f(grid[-1]) for f in fns])
    sc={'stored lp':lp,'native sksurv risk':native,'-S(last grid)':-last,'1-S(12)':1-S[:,0],'1-S(36)':1-S[:,1],'1-S(60)':1-S[:,2]}
    print(coh,'C-index by RSF risk score:',{k:round(concordance_index(d.survival_months,-v,d.os_status),4) for k,v in sc.items()})
    print(coh,'AUC(12,36,60) stored lp:',np.round(cumulative_dynamic_auc(r,y,lp,TIMES)[0],3),'| horizon-specific 1-S(h):',[round(cumulative_dynamic_auc(r,y,1-S[:,j],np.array([h]))[0][0],3) for j,h in enumerate(TIMES)])
