import sys,re,warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import docx
OUT='/mnt/user-data/outputs/'
plt.rcParams.update({'font.family':'Liberation Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COL={'Cox, spline age':'#1b6ca8','RSF':'#d95f02','Gradient-boosted':'#1b9e77','DeepSurv':'#7570b3','AJCC stage only':'#7f7f7f','Treat all':'#000000','Treat none':'#999999'}
NAME={'Cox, spline age':'Cox, spline age','RSF':'Random survival forest','Gradient-boosted':'Gradient-boosted Cox','DeepSurv':'DeepSurv','AJCC stage only':'AJCC stage only'}
# ---- Figure 1 flow ----
def box(ax,x,y,w,h,t,fs=9,fc='white',bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.01,rounding_size=0.01',fc=fc,ec='black',lw=0.9))
    ax.text(x,y,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.25)
def arr(ax,x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=0.9,color='black'))
fig,ax=plt.subplots(figsize=(11,9));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(0.03,0.975,'A. SEER',fontweight='bold',fontsize=11);ax.text(0.73,0.975,'B. TCGA-KIRC',fontweight='bold',fontsize=11)
xs=0.30
steps=[(0.925,'SEER Research Data, 17 Registries, Nov 2025 submission\n10,944,803 case records read'),
       (0.815,'Kidney and renal pelvis (site recode)\nn = 338,027'),
       (0.705,'Histologic type 8310 (clear cell adenocarcinoma)\nn = 153,805'),
       (0.595,'Diagnosed 2004–2017\nn = 88,248'),
       (0.485,'Malignant behavior (code 3), known age\nn = 88,246')]
for i,(y,t) in enumerate(steps):
    box(ax,xs,y,0.40,0.075,t)
    if i: arr(ax,xs,steps[i-1][0]-0.0375,xs,y+0.0375)
box(ax,0.30,0.37,0.40,0.075,'Analytic SEER cohort\nn = 87,924 (40,131 deaths)',bold=True,fc='#eeeeee');arr(ax,xs,0.485-0.0375,xs,0.37+0.0375)
box(ax,0.30,0.235,0.18,0.075,'Training set\nn = 61,546');box(ax,0.30+0.22,0.235,0.18,0.075,'Internal test set\nn = 26,378');box(ax,0.30-0.22,0.235,0.18,0.075,' ')
ax.patches[-1].remove();ax.texts[-1].remove()
arr(ax,0.30,0.37-0.0375,0.25,0.235+0.0375);arr(ax,0.30,0.37-0.0375,0.52,0.235+0.0375)
ax.patches[-3].set_x(0.21-0.09);ax.texts[-3].set_x(0.21)
for y,t in [(0.37+0.0,None)]: pass
# exclusions
box(ax,0.78-0.38,0.427,0.27,0.075,'Excluded (n = 322; 0.365%)\n151 missing survival time\n171 T, N, and M all missing',fs=8.5)
ax.annotate('',xy=(0.465,0.427),xytext=(0.30,0.427),arrowprops=dict(arrowstyle='-|>',lw=0.9))
ax.text(0.05,0.115,'Not possible with the extracted variables: removal of death-certificate-only or autopsy cases,\nsecond or later primary tumors, and patient-level deduplication (see Section 2.1).',fontsize=8.5,style='italic')
ax.text(0.05,0.065,'Zero-month survival recoded to 0.5 months: 650 (training) and 278 (test) patients.',fontsize=8.5,style='italic')
# TCGA
xt=0.82
box(ax,xt,0.925,0.30,0.075,'TCGA-KIRC, cBioPortal clinical data\nn = 512');box(ax,xt,0.785,0.30,0.075,'Excluded: missing M stage\nn = 2',fs=8.5)
arr(ax,xt,0.925-0.0375,xt,0.785+0.0375)
box(ax,xt,0.64,0.30,0.075,'External validation cohort\nn = 510 (170 deaths)',bold=True,fc='#eeeeee');arr(ax,xt,0.785-0.0375,xt,0.64+0.0375)
ax.text(xt,0.54,'Zero-month survival recoded: 2 patients\nN stage not assessed (NX): 265 patients',ha='center',fontsize=8.5,style='italic')
plt.savefig(OUT+'Figure1_flow_rebuilt.png',dpi=300,bbox_inches='tight');plt.savefig(OUT+'Figure1_flow_rebuilt.pdf',bbox_inches='tight');plt.close()
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
ax.set_xticks([0.5,1,2,4]);ax.set_xticklabels(['0.5','1','2','4']);ax.set_xlabel('Hazard ratio (95% CI), log scale')
for i in range(n): ax.text(7.5,n-1-i,f'{hr[i]:.2f} ({lo[i]:.2f}–{hi[i]:.2f})',va='center',fontsize=8)
ax.set_xlim(0.5,6);plt.tight_layout();plt.savefig(OUT+'Figure_forest_rebuilt.png',dpi=300);plt.savefig(OUT+'Figure_forest_rebuilt.pdf');plt.close()
# ---- Figure 4 calibration ----
d=pd.read_csv(OUT+'Calibration_deciles.csv');ms=['Cox, spline age','RSF','Gradient-boosted','DeepSurv']
fig,axs=plt.subplots(2,3,figsize=(11,7.2))
for i,coh in enumerate(['SEER','TCGA']):
    for j,y in enumerate([1,3,5]):
        ax=axs[i,j];g=d[(d.cohort==coh)&(d.year==y)];mx=max(g.mean_pred_risk.max(),g.obs_KM_risk.max())*1.05
        ax.plot([0,mx],[0,mx],color='black',lw=0.7,ls='--')
        for m in ms:
            h=g[g.model==m].sort_values('decile');ax.plot(h.mean_pred_risk,h.obs_KM_risk,'o-',ms=3.5,lw=1,color=COL[m],label=NAME[m])
        ax.set_xlim(0,mx);ax.set_ylim(0,mx);ax.set_xlabel('Mean predicted risk');ax.set_ylabel('Observed (Kaplan–Meier) risk')
        ax.set_title(f"{'ABCDEF'[i*3+j]}. {coh}, {y}-year",loc='left',fontsize=10)
        if i==0 and j==0: ax.legend(frameon=False,fontsize=8,loc='upper left')
plt.tight_layout();plt.savefig(OUT+'Figure_calibration_rebuilt.png',dpi=300);plt.savefig(OUT+'Figure_calibration_rebuilt.pdf');plt.close()
# ---- Figure 5 DCA ----
d=pd.read_csv(OUT+'DCA_all_strategies.csv');fig,axs=plt.subplots(1,2,figsize=(10,4.6))
for k,coh in enumerate(['SEER','TCGA']):
    ax=axs[k];g=d[(d.cohort==coh)&(d.year==5)]
    for s in ['Treat all','Treat none','AJCC stage only','Cox, spline age','RSF','Gradient-boosted','DeepSurv']:
        h=g[g.strategy==s].sort_values('threshold');ax.plot(h.threshold,h.net_benefit,color=COL[s],lw=1.6 if s in('Treat all','Treat none') else 1.2,ls='-' if s!='Treat none' else ':',label=NAME.get(s,s))
    ax.set_xlim(0,0.6);top=g[g.strategy!='Treat all'].net_benefit.max();ax.set_ylim(-0.05,top*1.08);ax.set_xlabel('Threshold probability of death by 5 years');ax.set_ylabel('Net benefit');ax.set_title(f"{'AB'[k]}. {coh}",loc='left')
    if k==0: ax.legend(frameon=False,fontsize=8)
plt.tight_layout();plt.savefig(OUT+'Figure_DCA_rebuilt.png',dpi=300);plt.savefig(OUT+'Figure_DCA_rebuilt.pdf');plt.close();print('done')
