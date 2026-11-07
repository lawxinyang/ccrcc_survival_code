# How the SEER data set was extracted

SEER data cannot be redistributed under the SEER Research Data Agreement. Approved users can rebuild the extract as follows.

## Database
SEER*Stat (version 9.1.0.2), database "Incidence – SEER Research Data, 17 Registries, Nov 2025 Sub (2000–2023)", released April 2026 (based on the November 2025 submission).

## Case selection
- Site recode ICD-O-3/WHO 2008 = Kidney and Renal Pelvis
- Histologic Type ICD-O-3 = 8310
- Year of diagnosis = 2004–2017
- Behavior code ICD-O-3 = Malignant

No other restriction was applied (no sequence number, reporting source, age, or diagnostic confirmation criteria). This extract yields 88,246 records.

## Variables exported
Sex; Race recode (W, B, AI, API); Vital status recode (study cutoff used); Survival months; COD to site recode; Grade Recode (thru 2017); Derived AJCC T, N, M and Stage Group, 6th ed (2004–2015); Derived AJCC Stage Group, T, N, M, 7th ed (2010–2015); 7th Edition Stage Group Recode (2016–2017); Derived SEER Combined T, N, M and Stage Group (2016–2017); Age recode with single ages and 90+; Laterality.
Not exported: patient ID, year of diagnosis, sequence number, primary site, type of reporting source, diagnostic confirmation. Deduplication, exclusion of second primaries, and exclusion of death-certificate-only and autopsy cases were therefore not possible.

## From the export to the model variables
Supplementary Table S2B of the paper gives the crosswalk from each SEER field to the model variables (age, sex, race, laterality, grade, T, N, M stage, AJCC stage group). Survival months and vital status give survival_months and os_status (1 = dead). Age is top-coded at 90 years. The script that converted the raw export into the cleaned file (88,246 rows, columns: age, sex, race, laterality, grade, T_stage, N_stage, M_stage, stage_group, survival_months, os_status, css_status) was not retained, so the crosswalk table is the specification.

## Cohort and split
`python scripts/make_cohort.py seer_kirc_cleaned.csv data/` removes 151 records without survival time and 171 records with T, N, and M all missing (87,924 remain) and makes the stratified 70/30 split (random_state = 42): 61,546 training and 26,378 test patients. Run on our cleaned file, this gives exactly the training and test files used in the paper.

## TCGA-KIRC
cBioPortal study kirc_tcga_pan_can_atlas_2018 (TCGA PanCancer Atlas), clinical data downloaded 31 July 2026; laterality from the Genomic Data Commons (project TCGA-KIRC, clinical.tsv, downloaded 31 July 2026); linked by case submitter ID (512 of 512 matched). Fields and mapping: Supplementary Table S2C. The cleaned file used in the paper is in data_public/ (512 rows; the 2 patients without M stage are excluded in the pipeline, leaving 510).
