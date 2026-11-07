import sys,re,warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import docx
OUT='/mnt/user-data/outputs/'
plt.rcParams.update({'font.family':'Liberation Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COL={'Cox, spline age':'#1b6ca8','RSF':'#d95f02','Gradient-boosted':'#1b9e77','DeepSurv':'#7570b3','AJCC stage only':'#7f7f7f','Treat all':'#000000','Treat none':'#999999'}
NAME={'Cox, spline age':'Cox, spline age','RSF':'Random survival forest','Gradient-boosted':'Gradient-boosted Cox','DeepSurv':'DeepSurv','AJCC stage only':'AJCC stage only'}
# ---- Figure 3 forest ----
src=docx.Document('/mnt/user-data/attachments/ca62109b-a916-5cd4-8db3-c8244fc66b01/6100d31f-Copy_of_Manuscript1.docx')
rows=[[c.text.strip() for c in r.cells] for r in src.tables[1].rows][1:]
lab=[];hr=[];lo=[];hi=[];sig=[]
for r in rows:
    lab.append(r[0]);hr.append(float(r[1]));a,b=r[2].replace('–','-').split('-');lo.append(float(a));hi.append(float(b));p=r[3];sig.append(p.startswith('<') or float(p)<0.05)
n=len(rows);fig,ax=plt.subplots(figsize=(8,7))
for i in range(n):
    y=n-1-i;c='black' if sig[i] else '#8a8a8a'
    ax.plot([lo[i],hi[i]],[y,y],color=c,lw=1.2);ax.plot(hr[i],y,'s',color=c,ms=5)
ax.axvline(1,color='black',lw=0.7,ls='--');ax.set_xscale('log');ax.set_yticks(range(n));ax.set_yticklabels(lab[::-1])
from matplotlib.ticker import NullFormatter;ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xticks([0.5,1,2,4]);ax.set_xticklabels(['0.5','1','2','4']);ax.set_xlabel('Hazard ratio (95% CI), log scale')
for i in range(n): ax.text(7.5,n-1-i,f'{hr[i]:.3f} ({lo[i]:.3f}–{hi[i]:.3f})',va='center',fontsize=8)
ax.set_xlim(0.5,6);plt.tight_layout();plt.savefig(OUT+'Figure_forest_rebuilt.png',dpi=300);plt.savefig(OUT+'Figure_forest_rebuilt.pdf');plt.close()
