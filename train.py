"""
Heart Disease Prediction - end-to-end ML pipeline
Run:  python train.py
Steps: load -> leakage audit -> EDA -> model comparison (repeated CV)
       -> tuning -> holdout evaluation -> interpretation -> save model
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold, cross_validate,
                                     GridSearchCV, StratifiedKFold, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score, brier_score_loss, classification_report)
warnings.filterwarnings("ignore")
SEED = 42
FIG = "figures/"
sns.set_theme(style="whitegrid", palette="Set2")

# ------------------------------------------------------------------ 1. LOAD
raw = pd.read_csv("data/heart.csv")
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
print(f"Raw shape: {raw.shape} | missing: {raw.isna().sum().sum()} | duplicates: {raw.duplicated().sum()}")

# ------------------------------------------------------------------ 2. LEAKAGE AUDIT
df = raw.drop_duplicates().reset_index(drop=True)
# LABEL AUDIT: in this file target=1 has HIGHER thalach and LOWER oldpeak/ca/exang (the healthy profile),
# i.e. the label is inverted vs clinical meaning. We define disease = 1 - target so metrics read correctly.
FLIP_LABELS = True
if FLIP_LABELS:
    df["target"] = 1 - df["target"]
print(df.groupby("target")[["thalach", "oldpeak", "ca", "exang"]].mean().round(2).rename(index={0: "no disease", 1: "disease"}))
X_all, y_all = raw.drop(columns="target"), raw["target"]
Xtr_l, Xte_l, ytr_l, yte_l = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
leaky = RandomForestClassifier(random_state=SEED).fit(Xtr_l, ytr_l)
leaky_acc = accuracy_score(yte_l, leaky.predict(Xte_l))
dup_in_test = pd.merge(Xte_l, Xtr_l.drop_duplicates(), how="inner").shape[0]
print(f"[Leakage] RF on raw split: acc={leaky_acc:.3f} (test rows with identical twin in train: {dup_in_test}/{len(Xte_l)})")
print(f"Deduplicated shape: {df.shape}")

# ------------------------------------------------------------------ 3. EDA
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
raw["target"].value_counts().sort_index().plot.bar(ax=ax[0], color=["#8da0cb", "#fc8d62"]); ax[0].set_title("Raw: class balance (original target)")
df["target"].value_counts().sort_index().plot.bar(ax=ax[1], color=["#8da0cb", "#fc8d62"]); ax[1].set_title("Deduplicated: class balance")
for a in ax: a.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(FIG + "01_class_balance.png", dpi=150); plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .7})
plt.title("Correlation matrix (deduplicated)"); plt.tight_layout(); plt.savefig(FIG + "02_correlation.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for a, c in zip(axes, NUM):
    sns.kdeplot(data=df, x=c, hue="target", fill=True, common_norm=False, ax=a)
plt.tight_layout(); plt.savefig(FIG + "03_numeric_distributions.png", dpi=150); plt.close()

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for a, c in zip(axes.ravel(), CAT):
    sns.barplot(data=df, x=c, y="target", ax=a, errorbar=None); a.set_ylabel("P(disease)"); a.set_title(c)
plt.tight_layout(); plt.savefig(FIG + "04_categorical_vs_target.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 4. SPLIT
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train {X_train.shape} | Test {X_test.shape}")

def prep(scale=True, ohe=True):
    num = StandardScaler() if scale else "passthrough"
    cat = OneHotEncoder(handle_unknown="ignore") if ohe else "passthrough"
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

models = {
    "Logistic Regression": Pipeline([("p", prep()), ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
    "SVM (RBF)":           Pipeline([("p", prep()), ("m", SVC(probability=True, random_state=SEED))]),
    "KNN":                 Pipeline([("p", prep()), ("m", KNeighborsClassifier())]),
    "Random Forest":       Pipeline([("p", prep(False, False)), ("m", RandomForestClassifier(n_estimators=400, random_state=SEED))]),
    "Gradient Boosting":   Pipeline([("p", prep(False, False)), ("m", GradientBoostingClassifier(random_state=SEED))]),
    "HistGradientBoosting":Pipeline([("p", prep(False, False)), ("m", HistGradientBoostingClassifier(random_state=SEED))]),
}

# ------------------------------------------------------------------ 5. MODEL COMPARISON
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1", "recall": "recall", "precision": "precision"}
rows, cv_raw = [], {}
for name, pipe in models.items():
    r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_raw[name] = r["test_roc_auc"]
    rows.append({"model": name, **{k: r[f"test_{k}"].mean() for k in scoring}, "auc_std": r["test_roc_auc"].std()})
cmp = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
print("\n=== Baseline CV comparison ===\n", cmp.round(3).to_string(index=False))
cmp.round(4).to_csv("outputs/model_comparison_baseline.csv", index=False)

plt.figure(figsize=(9, 5))
sns.boxplot(data=pd.DataFrame(cv_raw)[cmp["model"].tolist()], orient="h"); plt.xlabel("ROC-AUC (5x5 repeated CV)")
plt.title("Model comparison on training data"); plt.tight_layout(); plt.savefig(FIG + "05_model_comparison.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 6. TUNING
grids = {
    "Logistic Regression": {"m__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "SVM (RBF)": {"m__C": [0.3, 1, 3, 10], "m__gamma": ["scale", 0.01, 0.03, 0.1]},
    "Random Forest": {"m__max_depth": [3, 5, 8, None], "m__min_samples_leaf": [1, 3, 5], "m__max_features": ["sqrt", 0.5]},
    "Gradient Boosting": {"m__n_estimators": [50, 100, 200], "m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 3]},
}
inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
tuned = {}
for name, g in grids.items():
    gs = GridSearchCV(models[name], g, scoring="roc_auc", cv=inner, n_jobs=-1).fit(X_train, y_train)
    tuned[name] = gs
    print(f"Tuned {name:22s} CV AUC={gs.best_score_:.4f}  {gs.best_params_}")
best_name = max(tuned, key=lambda k: tuned[k].best_score_)
best = tuned[best_name].best_estimator_
print(f"\n>>> Selected model: {best_name}")

# ------------------------------------------------------------------ 7. HOLDOUT EVALUATION
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
def py(v): return v.item() if isinstance(v, np.generic) else v
metrics = {
    "selected_model": best_name,
    "best_params": {k: py(v) for k, v in tuned[best_name].best_params_.items()},
    "n_train": len(X_train), "n_test": len(X_test),
    "accuracy": accuracy_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba),
    "avg_precision": average_precision_score(y_test, proba), "f1": f1_score(y_test, pred),
    "recall_sensitivity": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
    "specificity": recall_score(y_test, pred, pos_label=0), "brier": brier_score_loss(y_test, proba),
    "leaky_raw_split_accuracy_for_comparison": leaky_acc,
}
rng = np.random.default_rng(SEED); aucs = []; yt = y_test.values
for _ in range(2000):
    i = rng.integers(0, len(yt), len(yt))
    if len(set(yt[i])) == 2: aucs.append(roc_auc_score(yt[i], proba[i]))
metrics["roc_auc_95ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
metrics = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}
metrics["roc_auc_95ci"] = [round(v, 4) for v in metrics["roc_auc_95ci"]]
print("\n=== Holdout metrics ===\n", json.dumps(metrics, indent=2))
print(classification_report(y_test, pred, target_names=["No disease", "Disease"]))
json.dump(metrics, open("outputs/metrics.json", "w"), indent=2)

# Screening threshold: highest threshold keeping >=90% recall on out-of-fold train predictions
oof = cross_val_predict(best, X_train, y_train, cv=inner, method="predict_proba")[:, 1]
ok = [t for t in np.linspace(0.05, 0.95, 91) if recall_score(y_train, oof >= t) >= 0.90]
thr = max(ok) if ok else 0.5
pred_s = (proba >= thr).astype(int)
screen = {"threshold": round(float(thr), 2), "recall": round(recall_score(y_test, pred_s), 4),
          "specificity": round(recall_score(y_test, pred_s, pos_label=0), 4),
          "precision": round(precision_score(y_test, pred_s), 4)}
print("Screening threshold (>=90% recall on CV):", screen)
json.dump(screen, open("outputs/screening_threshold.json", "w"), indent=2)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
ax[0].set_title("Confusion matrix (threshold 0.5)"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
fpr, tpr, _ = roc_curve(y_test, proba)
ax[1].plot(fpr, tpr, lw=2, label=f"AUC={metrics['roc_auc']:.3f}"); ax[1].plot([0, 1], [0, 1], "k--")
ax[1].set_title("ROC curve"); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate"); ax[1].legend()
p, r, _ = precision_recall_curve(y_test, proba)
ax[2].plot(r, p, lw=2, label=f"AP={metrics['avg_precision']:.3f}"); ax[2].set_title("Precision-Recall")
ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].legend()
plt.tight_layout(); plt.savefig(FIG + "06_holdout_performance.png", dpi=150); plt.close()

fp, mp = calibration_curve(y_test, proba, n_bins=5)
plt.figure(figsize=(5.5, 5)); plt.plot(mp, fp, "o-", label=best_name); plt.plot([0, 1], [0, 1], "k--", label="Perfect")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction"); plt.title("Calibration"); plt.legend()
plt.tight_layout(); plt.savefig(FIG + "07_calibration.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 8. INTERPRETATION
pi = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=50, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance")
imp.sort_values("importance", ascending=False).round(4).to_csv("outputs/feature_importance.csv", index=False)
plt.figure(figsize=(8, 6)); plt.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#66c2a5")
plt.xlabel("Drop in ROC-AUC when shuffled"); plt.title("Permutation feature importance (holdout)")
plt.tight_layout(); plt.savefig(FIG + "08_feature_importance.png", dpi=150); plt.close()
print("\nTop features:\n", imp.sort_values("importance", ascending=False).head(6).round(4).to_string(index=False))

# ------------------------------------------------------------------ 9. SAVE (refit on all deduplicated data)
final = tuned[best_name].best_estimator_.fit(X, y)
joblib.dump({"model": final, "features": list(X.columns), "screening_threshold": screen["threshold"]}, "models/heart_model.joblib")
print("\nSaved models/heart_model.joblib")
