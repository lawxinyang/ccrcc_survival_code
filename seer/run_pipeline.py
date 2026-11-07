"""Fit all models on the SEER training set and evaluate them in the SEER test set and TCGA-KIRC.
Usage:  CCRCC_DATA_DIR=/path/to/data python scripts/run_pipeline.py
Writes: models/ (fitted models, baselines, Cox coefficients) and results/table3_point_estimates.csv
Point estimates only; the bootstrap intervals of the paper (500 resamples SEER, 1,000 TCGA, seed 42; models not refitted) are not recomputed here."""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import numpy as np, pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from data import load, prep, design, design_spline
from mlmodels import GBCox, DeepSurv
from survival_tools import *
OUT = os.path.join(HERE, '..'); os.makedirs(os.path.join(OUT, 'models'), exist_ok=True); os.makedirs(os.path.join(OUT, 'results'), exist_ok=True)
tr, te, tc = load(); tr, te, tc = prep(tr), prep(te), prep(tc.dropna(subset=['M_stage']))   # TCGA: 2 patients without M stage excluded
print('n train/test/tcga:', len(tr), len(te), len(tc))
KN = np.quantile(tr.age, [.05, .35, .65, .95]); print('spline knots:', KN)
def stage(d):
    z = pd.get_dummies(d['stage_group']).astype(float); z.columns = [f'stage_group_{c}' for c in z.columns]; return z.drop(columns=['stage_group_I'], errors='ignore')
def inter(d):
    B = design_spline(d, KN)
    for pre in ['M_stage_', 'N_stage_', 'grade_']:
        for c in [c for c in B.columns if c.startswith(pre)]: B[f'age*{c}'] = B['age'] * B[c]
    for c in ['T_stage_T2', 'T_stage_T3', 'T_stage_T4']: B[f'age*{c}'] = B['age'] * B[c]
    return B
COX = {'AJCC stage only': stage, 'Cox, linear age': design, 'Cox, spline age': lambda d: design_spline(d, KN), 'Cox, spline + interactions': inter}
t, e = tr.survival_months.values, tr.os_status.values
models = {}; preds = {'SEER': {}, 'TCGA': {}}; coefs = []
for name, fx in COX.items():
    Xtr = fx(tr); cols = list(Xtr.columns); m = fit_cox(Xtr, tr); models[name] = (cols, m)
    ut, H = breslow_baseline(m.predict(Xtr), t, e)
    for coh, d in [('SEER', te), ('TCGA', tc)]:
        Xd = fx(d).reindex(columns=cols, fill_value=0.0); lp = m.predict(Xd); preds[coh][name] = (lp, surv_from_baseline(lp, ut, H))
    coefs += [{'model': name, 'term': c, 'coef': b} for c, b in zip(cols, m.coef_)]
# machine learning models: seven predictors, one-hot encoded
cols7 = list(design(tr).columns); mk = lambda d: design(d).reindex(columns=cols7, fill_value=0.0).values
Xtr, Xte, Xtc = mk(tr), mk(te), mk(tc)
t0 = time.time(); rsf = RandomSurvivalForest(n_estimators=105, max_depth=5, min_samples_leaf=50, max_features='sqrt', n_jobs=1, random_state=42).fit(Xtr, surv_obj(tr)); print('RSF', round(time.time() - t0), 's')
t0 = time.time(); gb = GBCox().fit(Xtr, t, e); print('GB', round(time.time() - t0), 's')
t0 = time.time(); ds = DeepSurv(seed=42).fit(Xtr, t, e); print('DeepSurv', round(time.time() - t0), 's')
base = {}
for nm, m in [('Gradient-boosted', gb), ('DeepSurv', ds)]:
    base[nm] = breslow_baseline(m.predict(Xtr), t, e)
    for coh, X_, d in [('SEER', Xte, te), ('TCGA', Xtc, tc)]:
        lp = m.predict(X_); preds[coh][nm] = (lp, surv_from_baseline(lp, *base[nm]))
rsf_native = {}
for coh, X_ in [('SEER', Xte), ('TCGA', Xtc)]:
    fn = rsf.predict_survival_function(X_, return_array=True); tt = rsf.unique_times_
    S = np.column_stack([fn[:, np.searchsorted(tt, h, side='right') - 1] for h in TIMES])
    preds[coh]['RSF'] = (rsf.predict(X_), S)            # C-index: ensemble mortality score; AUC and Brier: 1 - S(h)
# evaluation
ytr = surv_obj(tr); rows = []
for coh, d in [('SEER', te), ('TCGA', tc)]:
    ref = ytr if coh == 'SEER' else surv_obj(d)           # censoring distribution: SEER training set for SEER, TCGA itself for TCGA
    for name, (risk, S) in preds[coh].items():
        c, auc, br = evaluate(risk, S, d, ref)
        rows.append({'cohort': coh, 'model': name, 'C-index': round(c, 3), 'AUC 1y': round(auc[0], 3), 'AUC 3y': round(auc[1], 3), 'AUC 5y': round(auc[2], 3),
                     'Brier 1y': round(br[0], 3), 'Brier 3y': round(br[1], 3), 'Brier 5y': round(br[2], 3)})
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'results', 'table3_point_estimates.csv'), index=False); print(R.to_string(index=False))
pd.DataFrame(coefs).to_csv(os.path.join(OUT, 'models', 'cox_coefficients.csv'), index=False)
pickle.dump({'cox': models, 'rsf': rsf, 'gb': gb, 'ds': ds, 'cols7': cols7, 'knots': KN, 'baseline': base,
             'cox_baseline': {k: breslow_baseline(models[k][1].predict(COX[k](tr)), t, e) for k in COX}}, open(os.path.join(OUT, 'models', 'fitted_models.pkl'), 'wb'))
print('saved')
