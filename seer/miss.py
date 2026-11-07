import pandas as pd, numpy as np
D='/mnt/user-data/attachments/ca62109b-a916-5cd4-8db3-c8244fc66b01/'
r=pd.read_csv(D+'857d2ac9-seer_kirc_database.csv',low_memory=False); r.columns=[x.strip() for x in r.columns]
c=pd.read_csv(D+'aaebbd20-seer_kirc_cleaned.csv')
t=pd.read_csv(D+'81dfcc9e-kirc_tcga_pan_can_atlas_2018_clinical_data.csv',sep='\t',low_memory=False)
tc=pd.read_csv(D+'19f83181-tcga_kirc_cleaned.csv')
c['period']=np.where(r['Derived AJCC Stage Group, 6th ed (2004-2015)']=='Blank(s)','2016-2017','2004-2015')
excl=c.survival_months.isna()|c[['T_stage','N_stage','M_stage']].isna().any(axis=1)
a=c[~excl].copy(); print(len(c),len(a),excl.sum())
print('period counts raw',c.period.value_counts().to_dict(),'analytic',a.period.value_counts().to_dict())
print('age<18 analytic',(a.age<18).sum(),'age==90',(a.age==90).sum())
print('TCGA laterality',tc.laterality.value_counts(dropna=False).to_dict())
rows=[]
def add(ds,var,cats,d,unk):
    n=len(d)
    for k in cats:
        m=(d[var].isna()) if k=='<missing>' else (d[var]==k)
        rows.append([ds,var,k,int(m.sum()),round(100*m.sum()/n,2),'Yes' if k in unk else ''])
specs={'grade':(['G1','G2','G3','G4','GX'],['GX']),'T_stage':(['T1','T2','T3','T4','T0','TX','<missing>'],['T0','TX','<missing>']),
 'N_stage':(['N0','N+','NX','<missing>'],['NX','<missing>']),'M_stage':(['M0','M1','MX','<missing>'],['MX','<missing>']),
 'race':(['White','Black','Asian','Other/Unknown'],['Other/Unknown']),'laterality':(['Right','Left','Bilateral','Unknown'],['Unknown']),
 'stage_group':(['I','II','III','IV','Unknown','<missing>'],['Unknown','<missing>'])}
for ds,d in [('SEER raw export (n=88,246)',c),('SEER analysis cohort (n=87,924)',a),('SEER 2004-2015 analysis',a[a.period=='2004-2015']),('SEER 2016-2017 analysis',a[a.period=='2016-2017']),('TCGA cBioPortal (n=512)',tc),('TCGA analysis cohort (n=510)',tc.dropna(subset=['M_stage']))]:
    for v,(cats,unk) in specs.items():
        if ds.startswith('TCGA') and v=='stage_group' and False: pass
        add(ds,v,cats,d,unk)
M=pd.DataFrame(rows,columns=['Dataset','Variable','Category','n','%','Unknown/unassessed/missing category'])
M=M[M.n>0].reset_index(drop=True)
M.to_pickle('miss.pkl')
# compact view of unknown categories only
u=M[M['Unknown/unassessed/missing category']=='Yes']
print(u.pivot_table(index=['Variable','Category'],columns='Dataset',values='%').round(1).to_string())
