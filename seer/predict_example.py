"""Predicted survival at 1, 3, and 5 years for the patients in example_patients.csv (hypothetical patients, not real data).
Usage:  python example/predict_example.py [patients.csv]
The Cox, gradient-boosted, and DeepSurv models give S(t | x) = S0(t) ** exp(f(x)) with the Breslow baseline S0 estimated in the SEER
training set and frozen. The random survival forest gives its own ensemble survival function. Values are read from the right-continuous
step function without interpolation. These models were developed for patients after pathologic assessment; absolute risks were
underestimated in TCGA-KIRC (Section 3.7), so they are not for clinical use without recalibration."""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import numpy as np, pandas as pd
from data import design, design_spline
from survival_tools import TIMES, surv_from_baseline
M = pickle.load(open(os.path.join(HERE, '..', 'models', 'fitted_models.pkl'), 'rb'))
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'example_patients.csv'))
# the design matrices need every category level; pad with the training levels so a single patient is coded correctly
def align(fx_df, cols): return fx_df.reindex(columns=cols, fill_value=0.0)
KN = M['knots']
out = pd.DataFrame({'patient': range(1, len(df) + 1)})
cols, cox = M['cox']['Cox, spline age']; ut, H = M['cox_baseline']['Cox, spline age']
lp = cox.predict(align(design_spline(df, KN), cols)); S = surv_from_baseline(lp, ut, H)
for j, h in enumerate(TIMES): out[f'Cox spline: {int(h/12)}-y survival'] = S[:, j].round(3)
X7 = align(design(df), M['cols7']).values
for nm, mdl in [('Gradient-boosted', M['gb']), ('DeepSurv', M['ds'])]:
    lp = mdl.predict(X7); S = surv_from_baseline(lp, *M['baseline'][nm])
    for j, h in enumerate(TIMES): out[f'{nm}: {int(h/12)}-y survival'] = S[:, j].round(3)
fn = M['rsf'].predict_survival_function(X7, return_array=True); tt = M['rsf'].unique_times_
for j, h in enumerate(TIMES): out[f'RSF: {int(h/12)}-y survival'] = np.array([f[np.searchsorted(tt, h, side='right') - 1] for f in fn]).round(3)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(out.T.to_string(header=False))
out.to_csv(os.path.join(HERE, 'example_output.csv'), index=False)
