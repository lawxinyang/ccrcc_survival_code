import sys,pickle,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,'.')
from common import *;from gbclip import *
M=pickle.load(open('models_rebuilt.pkl','rb'));cols=M['cols'];tr_=prep(tr);te_=prep(te)
mk=lambda d:X(d).reindex(columns=cols,fill_value=0.0).values;Xtr,Xte=mk(tr_),mk(te_);t=tr_.survival_months.values;e=tr_.os_status.values
o=M['gb'].predict(Xte)
for c in [20,50]:
    g=GBCoxClip().fit(Xtr,t,e,clip=c);p=g.predict(Xte);print('clip',c,'corr',round(np.corrcoef(p,o)[0,1],5),'max abs diff',round(float(np.abs(p-o).max()),3),'range orig',o.min().round(2),o.max().round(2),'range clip',p.min().round(2),p.max().round(2),flush=True)
