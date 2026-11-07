import pandas as pd, numpy as np
D='/mnt/user-data/attachments/ca62109b-a916-5cd4-8db3-c8244fc66b01/'
tr=pd.read_csv(D+'9849bdcc-seer_train.csv'); te=pd.read_csv(D+'a1f56830-seer_test.csv')
tc=pd.read_csv(D+'19f83181-tcga_kirc_cleaned.csv')
REF={'sex':'Female','race':'White','grade':'G1','T_stage':'T1','N_stage':'N0','M_stage':'M0'}
def prep(df):
    d=df.copy(); d['survival_months']=d['survival_months'].where(d['survival_months']>0,0.5); return d
def X(df,vars_=('age','sex','race','grade','T_stage','N_stage','M_stage'),ref=REF):
    cols=[]; out=pd.DataFrame(index=df.index)
    for v in vars_:
        if v=='age': out['age']=df['age'].astype(float); continue
        lv=sorted(df[v].dropna().unique().tolist()) if v not in ('X',) else []
        r=ref.get(v) or ('Right' if v=='laterality' else lv[0])
        for l in lv:
            if l!=r: out[f'{v}_{l}']=(df[v]==l).astype(float)
    return out

def rcs_basis(x,knots):
    """Harrell restricted cubic spline: returns k-2 nonlinear columns."""
    k=np.asarray(knots,float); x=np.asarray(x,float); K=len(k)
    pos=lambda z:np.where(z>0,z,0.0)**3
    out=[]
    for j in range(K-2):
        t=(pos(x-k[j]) - pos(x-k[K-2])*(k[K-1]-k[j])/(k[K-1]-k[K-2]) + pos(x-k[K-1])*(k[K-2]-k[j])/(k[K-1]-k[K-2]))/(k[K-1]-k[0])**2
        out.append(t)
    return np.column_stack(out)
def X_spline(df,knots,vars_=('age','sex','race','grade','T_stage','N_stage','M_stage'),ref=REF):
    base=X(df,vars_,ref)
    nl=rcs_basis(df['age'].values,knots)
    for j in range(nl.shape[1]): base[f'age_rcs{j+1}']=nl[:,j]
    return base
