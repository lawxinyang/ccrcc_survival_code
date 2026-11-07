import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'Liberation Serif'})
OUT='/mnt/user-data/outputs/'
fig,ax=plt.subplots(figsize=(10,9));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
H=0.07
def box(x,y,w,t,fs=9.5,fc='white',bold=False,h=H):
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fc=fc,ec='black',lw=0.9))
    ax.text(x,y,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.3)
def arrow(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=0.9,color='black',shrinkA=0,shrinkB=0))
ax.text(0.03,0.985,'A. SEER',fontweight='bold',fontsize=12,va='top');ax.text(0.76,0.985,'B. TCGA-KIRC',fontweight='bold',fontsize=12,va='top')
xs=0.27;W=0.42
S=[(0.915,'SEER Research Data, 17 Registries, Nov 2025 submission\n10,944,803 case records read'),
   (0.805,'Site recode: kidney and renal pelvis\nn = 338,027'),
   (0.695,'Histologic type 8310 (clear cell adenocarcinoma)\nn = 153,805'),
   (0.585,'Year of diagnosis 2004–2017\nn = 88,248'),
   (0.475,'Malignant behavior (code 3), known age\nn = 88,246')]
for i,(y,t) in enumerate(S):
    box(xs,y,W,t)
    if i: arrow(xs,S[i-1][0]-H/2,xs,y+H/2)
ya=0.33
box(xs,ya,W,'Analytic SEER cohort\nn = 87,924 (40,131 deaths)',bold=True,fc='#eeeeee')
arrow(xs,0.475-H/2,xs,ya+H/2)
box(0.67,(0.475+ya)/2,0.30,'Excluded: n = 322 (0.365%)\n151 missing survival time\n171 T, N, and M all missing',fs=8.5,h=0.085)
arrow(xs,(0.475+ya)/2,0.67-0.15,(0.475+ya)/2)
yt=0.18
box(0.14,yt,0.22,'Training set\nn = 61,546');box(0.40,yt,0.22,'Internal test set\nn = 26,378')
ax.plot([xs,xs],[ya-H/2,ya-H/2-0.03],color='black',lw=0.9);ax.plot([0.14,0.40],[ya-H/2-0.03]*2,color='black',lw=0.9)
arrow(0.14,ya-H/2-0.03,0.14,yt+H/2);arrow(0.40,ya-H/2-0.03,0.40,yt+H/2)
ax.text(0.03,0.075,'Patient-level deduplication, identification of second or later primary tumors, and exclusion of death-certificate-only\nor autopsy-only cases were not possible with the extracted variables (Section 2.1).\nZero-month survival times recoded to 0.5 months: 650 (training) and 278 (test) patients.',fontsize=8.5,style='italic',va='center',linespacing=1.4)
xt=0.865;Wt=0.24
box(xt,0.915,Wt,'cBioPortal clinical data\nn = 512')
box(xt,0.805,Wt,'Excluded: missing M stage\nn = 2',fs=8.5)
arrow(xt,0.915-H/2,xt,0.805+H/2)
box(xt,0.66,Wt,'External validation cohort\nn = 510 (170 deaths)',bold=True,fc='#eeeeee')
arrow(xt,0.805-H/2,xt,0.66+H/2)
ax.text(xt,0.575,'Zero-month survival recoded: 2\nN stage not assessed (NX): 265',ha='center',va='center',fontsize=8.5,style='italic',linespacing=1.4)
plt.savefig(OUT+'Figure1_flow_rebuilt.png',dpi=300,bbox_inches='tight');plt.savefig(OUT+'Figure1_flow_rebuilt.pdf',bbox_inches='tight')
