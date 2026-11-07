# Code for "Explainable Machine Learning for Overall Survival Prediction in Clear Cell Renal Cell Carcinoma"

Code for the revised manuscript (Array, ARRAY-D-26-05730).

**Important.** The original analysis scripts and fitted models of the first submission were not retained. All models were refitted during the revision with the code in this repository, and every number in the revised manuscript comes from it. The code reproduces the point estimates of Table 3 (98 values: C-index, 1-, 3-, and 5-year AUC and Brier score for seven models in two cohorts) exactly to three decimals. It does not reproduce three TCGA values of the first submission (Supplementary Table S1). `scripts/table2_cox_hazard_ratios.py` reproduces Table 2 (hazard ratios of the Cox model with a linear age term, Efron ties) and Figure 3.

## Contents
| Folder | What it holds |
|---|---|
| `src/` | Data preparation (`data.py`), Cox partial-likelihood pieces (`coxcore.py`), gradient-boosted Cox model and DeepSurv (`mlmodels.py`), Breslow baseline and metrics (`survival_tools.py`), exact SHAP (`shap_exact.py`) |
| `scripts/run_pipeline.py` | Fits all models on the SEER training set and evaluates them in the SEER test set and TCGA-KIRC; writes `results/table3_point_estimates.csv`; the fitted models are written to `models/` on your own machine |
| `scripts/table2_cox_hazard_ratios.py` | Table 2 (hazard ratios, 95% CI, P) and Figure 3 (forest plot); writes `results/table2_hazard_ratios.csv` |
| `scripts/make_cohort.py` | Builds the analysis cohort and the 70/30 split from the cleaned SEER file |
| `models/` | Not included. Section 19 of the SEER Research Data Agreement does not allow models developed with the data, including their parameters, to be shared except with other approved SEER users. This folder is created when you run `scripts/run_pipeline.py` on your own approved SEER extract |
| `example/` | `predict_example.py` and three hypothetical patients; it needs the models from your own run of the pipeline, and no output is included |
| `seer_extraction/` | How to rebuild the SEER extract with SEER*Stat |
| `data_public/` | TCGA-KIRC file used in the paper (public data) |
| `as_run_analysis_scripts/` | All scripts that produced the tables and figures of the revision, as they were run (not a one-command pipeline; see the README in that folder) |

## Data
- **SEER**: not included (SEER Research Data Agreement). Follow `seer_extraction/SEER_extraction_instructions.md`.
- **TCGA-KIRC**: `data_public/tcga_kirc_cleaned.csv` (cBioPortal study `kirc_tcga_pan_can_atlas_2018` with laterality from the GDC; downloaded 31 July 2026).

## Environment
Python 3.13; `pip install -r requirements.txt` (numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.9.1, scikit-survival 0.28.0, lifelines 0.30.3, matplotlib 3.11.2).

## Reproduce
```
pip install -r requirements.txt
python scripts/make_cohort.py seer_kirc_cleaned.csv data/        # needs your own SEER extract
cp data_public/tcga_kirc_cleaned.csv data/
CCRCC_DATA_DIR=data python scripts/run_pipeline.py                 # about 1 minute
```
Seeds: split 42; random survival forest 42; DeepSurv 42; gradient-boosted model has no subsampling. Settings: random survival forest 105 trees, depth 5, minimum leaf 50, √p features (scikit-survival); gradient-boosted Cox 216 rounds, depth 3, learning rate 0.1, no row subsampling; DeepSurv 8 ReLU units, full-batch Adam, learning rate 0.02, 531 epochs, standardized inputs. Cox models use Efron ties; the Breslow baseline is estimated once in the SEER training set and frozen. Spline knots: 40, 57, 67, 81 years (5th, 35th, 65th, 95th percentiles of training age).

## Prediction example
```
python example/predict_example.py                 # three hypothetical patients
python example/predict_example.py my_patients.csv # same columns as example/example_patients.csv
```
It prints predicted survival at 1, 3, and 5 years from the spline Cox model, the gradient-boosted model, DeepSurv, and the random survival forest. It requires the models written by `scripts/run_pipeline.py`, so run the pipeline on your own approved SEER extract first. The models are least reliable for rare, extreme combinations of predictors.

## Limits of use
The models were developed for patients after pathologic assessment (the predictors come from surgical pathology) and predict all-cause mortality from diagnosis. In TCGA-KIRC the models underestimated 5-year mortality (observed-to-expected ratio 1.23–1.31), so absolute risks should not be quoted to patients without recalibration in the target population. This code is for research and reproduction, not for clinical use.

## Notes
- The fitted models are created by `scripts/run_pipeline.py` and saved as a Python pickle in `models/`; load such a file only from a source you trust, with the library versions above. Do not publish or share these files: they are models trained on SEER data (Section 19 of the SEER Research Data Agreement).
- Bootstrap intervals, calibration, decision curves, SHAP, and the other analyses of the paper are in `as_run_analysis_scripts/`; they were not rewritten into the single pipeline.
- License: MIT for the code (see `LICENSE`).

## Citation
[CONFIRM: add the citation of the article and the Zenodo DOI after deposit]
