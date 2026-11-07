"""From the cleaned SEER file (one row per SEER record, 88,246 rows, model variables already mapped; see Supplementary Table S2B)
to the analysis cohort and the training/test split.
Usage:  python scripts/make_cohort.py seer_kirc_cleaned.csv OUTPUT_DIR
Steps (Methods 2.1 and 2.3): 1) drop records with missing survival time (151); 2) drop records with T, N, and M all missing (171);
3) stratified 70/30 split by death status, random_state = 42  ->  seer_train.csv (61,546) and seer_test.csv (26,378)."""
import sys, os, pandas as pd
from sklearn.model_selection import train_test_split
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = pd.read_csv(src); print('records:', len(d))
d = d[d.survival_months.notna()]; print('after missing survival time removed:', len(d))
d = d[~d[['T_stage', 'N_stage', 'M_stage']].isna().all(axis=1)]; print('after T, N, M all missing removed:', len(d))
tr, te = train_test_split(d, test_size=0.3, stratify=d['os_status'], random_state=42)
tr.to_csv(os.path.join(out, 'seer_train.csv'), index=False); te.to_csv(os.path.join(out, 'seer_test.csv'), index=False); print('train/test:', len(tr), len(te))
