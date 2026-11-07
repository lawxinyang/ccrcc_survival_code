"""Data preparation. Set CCRCC_DATA_DIR to the folder holding seer_train.csv, seer_test.csv and tcga_kirc_cleaned.csv
(columns: age, sex, race, laterality, grade, T_stage, N_stage, M_stage, stage_group, survival_months, os_status)."""
import os, numpy as np, pandas as pd
REF = {'sex': 'Female', 'race': 'White', 'grade': 'G1', 'T_stage': 'T1', 'N_stage': 'N0', 'M_stage': 'M0'}
SEVEN = ('age', 'sex', 'race', 'grade', 'T_stage', 'N_stage', 'M_stage')
def load(data_dir=None):
    d = data_dir or os.environ.get('CCRCC_DATA_DIR', '.')
    return (pd.read_csv(os.path.join(d, 'seer_train.csv')), pd.read_csv(os.path.join(d, 'seer_test.csv')),
            pd.read_csv(os.path.join(d, 'tcga_kirc_cleaned.csv')))
def prep(df):
    """Zero-month survival times are recoded to 0.5 months."""
    d = df.copy(); d['survival_months'] = d['survival_months'].where(d['survival_months'] > 0, 0.5); return d
def design(df, vars_=SEVEN, ref=REF):
    """Dummy coding with fixed reference categories; levels are taken from the data passed in (align columns afterwards)."""
    out = pd.DataFrame(index=df.index)
    for v in vars_:
        if v == 'age': out['age'] = df['age'].astype(float); continue
        lv = sorted(df[v].dropna().unique().tolist())
        r = ref.get(v) or ('Right' if v == 'laterality' else lv[0])
        for l in lv:
            if l != r: out[f'{v}_{l}'] = (df[v] == l).astype(float)
    return out
def rcs_basis(x, knots):
    """Harrell restricted cubic spline: returns the k-2 nonlinear columns."""
    k = np.asarray(knots, float); x = np.asarray(x, float); K = len(k)
    pos = lambda z: np.where(z > 0, z, 0.0) ** 3
    return np.column_stack([(pos(x - k[j]) - pos(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
            + pos(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / (k[K - 1] - k[0]) ** 2 for j in range(K - 2)])
def design_spline(df, knots, vars_=SEVEN, ref=REF):
    B = design(df, vars_, ref); nl = rcs_basis(df['age'].values, knots)
    for j in range(nl.shape[1]): B[f'age_rcs{j+1}'] = nl[:, j]
    return B
