These are the scripts that produced the tables, figures, and supplementary results of the revised manuscript, as they were run during the
revision. They were written and run step by step, they read intermediate files (pickles, CSV) written by earlier scripts, and some contain
absolute paths from the working environment, so they are NOT a one-command pipeline. They are included for transparency. Several were
superseded by later versions (for example table3.py by table3final.py and fixrsf.py, gbt.py and gbt2.py by mlmodels.py and gbclip.py).
The runnable pipeline is in ../scripts/ and reproduces the point estimates of Table 3.
Script groups: bootstrap and paired differences (boot*.py, consol*.py, fullci.py, fixrsf.py); calibration (calib.py); decision curves (dca2.py, dca3.py);
proportional hazards (ph*.py, cox1.py, ../ph_global.py); calendar-period analysis (period.py); censoring and follow-up (r111.py, kmtert.py);
SHAP (shap*.py); sensitivity analyses (complete.py, norace*.py, variants.py, inter.py); grid search (grid.py); LASSO (lasso.py); figures (fig*.py, figs*.py).
