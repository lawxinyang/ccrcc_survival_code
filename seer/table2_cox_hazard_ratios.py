"""Table 2 and Figure 3: multivariable Cox model with a linear age term in the SEER training set (Efron ties).
Usage:  CCRCC_DATA_DIR=/path/to/data python scripts/table2_cox_hazard_ratios.py
Writes: results/table2_hazard_ratios.csv (full precision) and results/Figure_3_forest.png / .pdf
Zero-month survival times are recoded to 0.5 months, as in the main pipeline (src/data.py)."""
import os, sys, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from lifelines import CoxPHFitter
from data import load, prep
tr = prep(load()[0]); OUT = os.path.join(HERE, '..', 'results'); os.makedirs(OUT, exist_ok=True)
X = pd.DataFrame({'age': tr.age, 'sex_Male': (tr.sex == 'Male') * 1})
for col, ref in [('race', 'White'), ('grade', 'G1'), ('T_stage', 'T1'), ('N_stage', 'N0'), ('M_stage', 'M0')]:
    for lev in sorted(tr[col].unique()):
        if lev != ref: X[f'{col}_{lev}'] = (tr[col] == lev) * 1
X['t'] = tr.survival_months; X['e'] = tr.os_status
s = CoxPHFitter().fit(X, 't', 'e').summary   # lifelines uses Efron's method for tied event times
order = [('age', 'Age (per 1-y increase)'), ('sex_Male', 'Sex: Male vs. Female'), ('race_Black', 'Race: Black vs. White'), ('race_Asian', 'Race: Asian vs. White'),
         ('race_Other/Unknown', 'Race: Other/Unknown vs. White'), ('grade_G2', 'Grade: G2 vs. G1'), ('grade_G3', 'Grade: G3 vs. G1'), ('grade_G4', 'Grade: G4 vs. G1'),
         ('grade_GX', 'Grade: GX vs. G1'), ('T_stage_T2', 'T stage: T2 vs. T1'), ('T_stage_T3', 'T stage: T3 vs. T1'), ('T_stage_T4', 'T stage: T4 vs. T1'),
         ('T_stage_T0', 'T stage: T0 vs. T1'), ('T_stage_TX', 'T stage: TX vs. T1'), ('N_stage_N+', 'N stage: N+ vs. N0'), ('N_stage_NX', 'N stage: NX vs. N0'),
         ('M_stage_M1', 'M stage: M1 vs. M0'), ('M_stage_MX', 'M stage: MX vs. M0')]
tab = pd.DataFrame({'variable': [l for _, l in order], 'coef': [s.loc[k, 'coef'] for k, _ in order], 'se': [s.loc[k, 'se(coef)'] for k, _ in order],
                    'HR': [s.loc[k, 'exp(coef)'] for k, _ in order], 'CI_low': [s.loc[k, 'exp(coef) lower 95%'] for k, _ in order],
                    'CI_high': [s.loc[k, 'exp(coef) upper 95%'] for k, _ in order], 'P': [s.loc[k, 'p'] for k, _ in order]})
tab.to_csv(os.path.join(OUT, 'table2_hazard_ratios.csv'), index=False)
print(tab.round({'HR': 3, 'CI_low': 3, 'CI_high': 3, 'P': 4}).to_string(index=False)); print('n =', len(X), 'deaths =', int(X.e.sum()))
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
n = len(tab); fig, ax = plt.subplots(figsize=(8, 7))
for i, r in tab.iterrows():
    y = n - 1 - i; c = 'black' if r.P < 0.05 else '#8a8a8a'
    ax.plot([r.CI_low, r.CI_high], [y, y], color=c, lw=1.2); ax.plot(r.HR, y, 's', color=c, ms=5)
    ax.text(7.5, y, f'{r.HR:.3f} ({r.CI_low:.3f}–{r.CI_high:.3f})', va='center', fontsize=8)
ax.axvline(1, color='black', lw=0.7, ls='--'); ax.set_xscale('log'); ax.set_yticks(range(n)); ax.set_yticklabels(tab.variable[::-1])
ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(['0.5', '1', '2', '4']); ax.set_xlabel('Hazard ratio (95% CI), log scale'); ax.set_xlim(0.5, 6)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'Figure_3_forest.png'), dpi=300); plt.savefig(os.path.join(OUT, 'Figure_3_forest.pdf')); plt.close()
