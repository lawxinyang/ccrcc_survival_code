import pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
D='/mnt/user-data/attachments/ca62109b-a916-5cd4-8db3-c8244fc66b01/'
r=pd.read_csv(D+'857d2ac9-seer_kirc_database.csv',low_memory=False); r.columns=[x.strip() for x in r.columns]
c=pd.read_csv(D+'aaebbd20-seer_kirc_cleaned.csv')
t=pd.read_csv(D+'81dfcc9e-kirc_tcga_pan_can_atlas_2018_clinical_data.csv',sep='\t',low_memory=False)
tc=pd.read_csv(D+'19f83181-tcga_kirc_cleaned.csv')
c['period']=np.where(r['Derived AJCC Stage Group, 6th ed (2004-2015)']=='Blank(s)','2016-2017','2004-2015')
excl=c.survival_months.isna()|c[['T_stage','N_stage','M_stage']].isna().any(axis=1)
a=c[~excl]
# 1 missingness (exact percentages, no double rounding)
specs={'grade':['G1','G2','G3','G4','GX'],'T_stage':['T1','T2','T3','T4','T0','TX'],'N_stage':['N0','N+','NX'],'M_stage':['M0','M1','MX'],
 'race':['White','Black','Asian','Other/Unknown'],'laterality':['Right','Left','Bilateral','Unknown'],'stage_group':['I','II','III','IV','Unknown']}
unk={'GX','T0','TX','NX','MX','Other/Unknown','Unknown'}
dsets=[('SEER raw export',c),('SEER analysis cohort',a),('SEER 2004-2015 (analysis)',a[a.period=='2004-2015']),('SEER 2016-2017 (analysis)',a[a.period=='2016-2017']),('TCGA cBioPortal',tc),('TCGA analysis cohort',tc.dropna(subset=['M_stage']))]
miss=[]
for v,cats in specs.items():
    for k in cats+['<blank>']:
        row=[v,k,'unknown / not assessed' if k in unk else ('missing' if k=='<blank>' else '')]
        tot=0
        for nm,d in dsets:
            n=len(d); x=d[v].isna().sum() if k=='<blank>' else (d[v]==k).sum()
            tot+=x; row+= [int(x),f'{100*x/n:.1f}']
        if tot>0: miss.append(row)
hdr=['Variable','Category','Type']
for nm,d in dsets: hdr+=[f'{nm}\n(n={len(d):,})\nn','%']
# 2 SEER crosswalk
sc=[
['Age','Age recode with single ages and 90+','"N years" -> integer; "90+ years" -> 90 (top-coded)','Continuous (years)','259 analysis patients coded 90; 14 aged <18'],
['Sex','Sex','Male / Female','Male / Female',''],
['Race','Race recode (W, B, AI, API)','White; Black; Asian or Pacific Islander','White; Black; Asian','American Indian/Alaska Native (1,057) and Unknown (433) merged'],
['','','American Indian/Alaska Native; Unknown','Other/Unknown','n=1,490 in raw export'],
['Laterality','Laterality','Right - origin of primary; Left - origin of primary; Bilateral, single primary','Right; Left; Bilateral','Not used in final model (LASSO)'],
['','','Only one side - side unspecified (59); Paired site, no laterality information (276)','Unknown',''],
['Grade','Grade Recode (thru 2017)','Well / Moderately / Poorly differentiated; Undifferentiated, anaplastic','G1; G2; G3; G4','SEER recode that pools source grading systems; see note'],
['','','Unknown','GX','12,793 in raw export'],
['T stage (2004-2015)','Derived AJCC T, 6th ed (2004-2015)','T1, T1NOS, T1a, T1b','T1',''],
['','','T2','T2',''],
['','','T3, T3NOS, T3a, T3b, T3c','T3',''],
['','','T4; T0; TX','T4; T0; TX','T0 and TX kept as separate levels (pooled as T0/TX in Table 1)'],
['T stage (2016-2017)','Derived SEER Combined T (2016-2017)','c1/c1A/c1B/p1/p1A/p1B','T1','Clinical (c) and pathological (p) prefixes pooled'],
['','','c2/c2A/c2B/p2/p2A/p2B','T2',''],
['','','c3/c3A/c3B/c3C/p3/p3A/p3B/p3C','T3',''],
['','','c4/p4; c0 (n=5); cX/pX','T4; T0; TX',''],
['N stage (2004-2015)','Derived AJCC N, 6th ed (2004-2015)','N0 / N1, N2 / NX','N0 / N+ / NX',''],
['N stage (2016-2017)','Derived SEER Combined N (2016-2017)','c0, p0 / c1, p1 / cX, pX','N0 / N+ / NX','Clinical and pathological pooled'],
['M stage (2004-2015)','Derived AJCC M, 6th ed (2004-2015)','M0 / M1 / MX','M0 / M1 / MX',''],
['M stage (2016-2017)','Derived SEER Combined M (2016-2017)','c0 / c1, p1','M0 / M1','No MX category in this field'],
['AJCC stage group (benchmark only)','Derived AJCC Stage Group, 6th ed (2004-2015); Derived SEER Cmb Stg Grp (2016-2017)','I-IV; "UNK Stage" or 99','I; II; III; IV; Unknown','Unknown retained as its own level in the stage-only benchmark'],
['Exclusion 1','Survival months','missing (151)','Excluded',''],
['Exclusion 2','Derived T, N, M (both periods)','all three missing (175; 4 already excluded under 1)','Excluded (171 additional)','Total excluded 322 = 0.365% of 88,246'],
['Outcome','Vital status recode (study cutoff used)','Dead / Alive','os_status 1 / 0','Death from any cause'],
['Survival time','Survival months','0 months (928 patients in the analysis cohort)','recoded to 0.5',''],
]
tcx=[
['Age','Diagnosis Age','integer years','Continuous',''],
['Sex','Sex','Male / Female','Male / Female',''],
['Race','Race Category','White (442); Black or African American (55); Asian (8)','White; Black; Asian','n shown for the 512-patient file'],
['','','blank (7)','Other/Unknown','7 patients with no recorded race'],
['Grade','Neoplasm Histologic Grade','G1 / G2 / G3 / G4','G1 / G2 / G3 / G4',''],
['','','GX (5); blank (3)','GX','3 blank grades recoded to GX'],
['T stage','American Joint Committee on Cancer Tumor Stage Code','T1, T1A, T1B','T1',''],
['','','T2, T2A, T2B','T2',''],
['','','T3, T3A, T3B, T3C','T3',''],
['','','T4','T4','No T0 or TX in TCGA'],
['N stage','Neoplasm Disease Lymph Node Stage AJCC Code','N0 / N1 / NX','N0 / N+ / NX','NX 267 of 512'],
['M stage','AJCC Metastasis Stage Code','M0 / M1 / MX','M0 / M1 / MX','29 MX kept as a category'],
['','','blank (2)','Excluded','The only TCGA exclusion; differs from MX (assessed as not evaluable)'],
['AJCC stage group (benchmark only)','Neoplasm Disease Stage AJCC Code','STAGE I-IV','I; II; III; IV','No unknown stage'],
['Laterality','GDC clinical.tsv (merged by case)','Right / Left / Bilateral','Right (266); Left (242); Bilateral (4)','Not used in final model'],
['Outcome','Overall Survival Status','0:LIVING / 1:DECEASED','os_status 0 / 1',''],
['Survival time','Overall Survival (Months)','months from initial pathologic diagnosis (cBioPortal definition)','survival_months','2 patients with 0 months recoded to 0.5'],
['Study','Study ID','kirc_tcga_pan_can_atlas_2018','','512 unique Patient IDs, 512 primary tumor samples, 1 sample per patient (no deduplication needed)'],
['Histology','ICD-O-3 Histology Code','8310/3 (499); 8312/3 (13)','Not restricted','13 patients are coded 8312/3 (renal cell carcinoma, NOS) while SEER is restricted to 8310'],
['AJCC edition','AJCC Publication Version Type','7TH (90); 6TH (72); 5TH (5); blank (345)','Not harmonized','Edition is missing for 67% of patients'],
['Neoadjuvant therapy','Neoadjuvant Therapy Type Administered Prior To Resection','Yes (9); No (503)','Not restricted',''],
]
pc=pd.DataFrame({'Item':['Diagnoses 2004-2015 (6th-edition derived fields populated)','Diagnoses 2016-2017 (6th-edition fields blank; SEER Combined fields populated)','Patients with 7th-edition (2010-2015) fields populated','Excluded: missing survival time','Excluded: T, N and M all missing (beyond the above)','Zero-month survival, SEER analysis cohort (train 650, test 278)','Zero-month survival, TCGA analysis cohort'],
 'n':[int((c.period=='2004-2015').sum()),int((c.period=='2016-2017').sum()),int((r['Derived AJCC T, 7th ed (2010-2015)']!='Blank(s)').sum()),151,171,928,2]})
wb=Workbook(); thin=Side(style='thin'); 
def sheet(ws,header,rows,widths,title=None):
    ws.append(header)
    for x in rows: ws.append(x)
    for cell in ws[1]:
        cell.font=Font(name='Times New Roman',bold=True); cell.alignment=Alignment(wrap_text=True,vertical='center'); cell.border=Border(top=thin,bottom=thin)
    for row in ws.iter_rows(min_row=2):
        for cell in row: cell.font=Font(name='Times New Roman'); cell.alignment=Alignment(wrap_text=True,vertical='top')
    for i,w in enumerate(widths,1): ws.column_dimensions[get_column_letter(i)].width=w
    ws.freeze_panes='A2'
ws=wb.active; ws.title='Missingness'; sheet(ws,hdr,miss,[16,16,20]+[11,7]*6)
sheet(wb.create_sheet('SEER crosswalk'),['Model variable','SEER field (original name)','Raw values','Model category','Note'],sc,[24,42,50,28,50])
sheet(wb.create_sheet('TCGA crosswalk'),['Model variable','cBioPortal / GDC field','Raw values','Model category','Note'],tcx,[24,44,48,30,56])
sheet(wb.create_sheet('Counts'),['Item','n'],pc.values.tolist(),[78,10])
wb.save('/mnt/user-data/outputs/R1-8_R1-9_缺失值表与变量对照表.xlsx')
u=pd.DataFrame(miss,columns=['v','k','type']+[f'{i}{j}' for i in range(6) for j in 'np'])
print(u.iloc[:, [0,1,3+2*1+1,3+2*2+1,3+2*3+1,3+2*5+1]].to_string())
