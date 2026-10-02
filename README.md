![Heart Disease Prediction](assets/heart-disease-banner.png)
# Heart Disease Prediction — End-to-End ML Project

Predicts whether a patient has heart disease from 13 clinical features (UCI Cleveland data, Kaggle copy).

## The key finding: data leakage
The provided `heart.csv` has **1,025 rows but only 302 unique patients** (723 exact duplicates).
A standard random split puts identical twins in train and test:

| Evaluation | Random Forest accuracy |
|---|---|
| Naive split on raw data (202 of 205 test rows have a twin in train) | **100%** — meaningless, memorisation |
| Deduplicated, proper holdout (this project) | **~82%** — realistic |

Any tutorial showing ~99-100% on this file is reporting leakage, not skill.

## Pipeline (`train.py`)
1. Load + audit (missing values, duplicates, leakage demo)
2. EDA figures (balance, correlations, distributions, categorical effects)
3. Deduplicate → stratified 80/20 split (241 train / 61 test)
4. Leak-free `Pipeline`s (scaling / one-hot fitted inside CV folds only)
5. Compare 6 models with **5×5 repeated stratified CV** (ROC-AUC, accuracy, F1, recall, precision)
6. GridSearchCV tuning of the top candidates
7. One-time holdout evaluation: ROC, PR, calibration, confusion matrix, bootstrap CI
8. Permutation importance (model-agnostic interpretation)
9. Screening threshold chosen on CV predictions (≥90% recall target)
10. Refit on all unique data → `models/heart_model.joblib`

## Second finding: the label is inverted
In this file `target=1` patients have *higher* max heart rate and *lower* ST depression, fewer blocked vessels and less exercise angina,
the clinically healthy profile. The pipeline therefore defines `disease = 1 - target` (`FLIP_LABELS = True` in `train.py`;
set it to `False` if you know your copy is coded the other way) so every metric refers to the true disease class.

## Results
| Model (tuned, 5-fold CV ROC-AUC) | Score |
|---|---|
| **Random Forest** (selected) | **0.916** |
| Logistic Regression | 0.915 |
| SVM (RBF) | 0.915 |
| Gradient Boosting | 0.911 |

The top three are statistically tied (CV std ≈ 0.04), so the choice is a near coin-flip; logistic regression is the simpler, more interpretable alternative.

**Holdout (61 patients):** accuracy 0.820 · ROC-AUC 0.889 (95% CI 0.80–0.96) · recall 0.786 · specificity 0.848 · precision 0.815 · Brier 0.137
**Screening mode** (threshold 0.35, tuned for ≥90% recall in CV): recall 0.893, specificity 0.636, precision 0.676.

## Top predictors (permutation importance)
`cp` (chest-pain type), `thal`, `ca`, `thalach`, `sex`, `oldpeak`.

## Limitations (be honest in your viva / interview)
- Only 302 unique patients; the holdout of 61 is noisy (AUC CI is wide).
- Single-hospital, 1980s data — not validated for real clinical use.
- `ca` and `thal` come from invasive/nuclear tests, so this is not a cheap first-line screener.
- Educational project, **not medical advice**.

## Run
```bash
pip install -r requirements.txt
python train.py      # regenerates figures/, models/, outputs/
python predict.py    # demo prediction
```

## Structure
```
data/heart.csv  train.py  predict.py  requirements.txt
figures/  (8 PNGs)   models/heart_model.joblib   outputs/  (metrics.json, CSVs)
```

## Ideas to extend
Add SHAP, a Streamlit/Gradio web app, external validation on the Statlog/Hungarian/Swiss UCI sets, calibration with `CalibratedClassifierCV`, cost-sensitive threshold tuning.
