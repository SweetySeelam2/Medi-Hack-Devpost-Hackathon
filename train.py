import os, json, joblib
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
)
import matplotlib.pyplot as plt

# ---- paths/seed
DATA = Path("data/heart.csv")
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# ---- load
df = pd.read_csv(DATA)
y = df["target"].astype(int)
X = df.drop(columns=["target"])

# identify columns
cat_cols = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
num_cols = [c for c in X.columns if c not in cat_cols]

# ---- train/val/test split: keep a small val set for calibration only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
)

# ---- preprocessing
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ---- candidates
pipe_lr = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
])
grid_lr = {
    "clf__C": [0.1, 1.0, 3.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"]
}

pipe_rf = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
])
grid_rf = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 6, 10],
    "clf__min_samples_leaf": [1, 2]
}

# ---- cross-validated model selection on X_tr
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
def cv_best(model, grid):
    gs = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs

gs_lr = cv_best(pipe_lr, grid_lr)
gs_rf = cv_best(pipe_rf, grid_rf)

best_gs = gs_lr if gs_lr.best_score_ >= gs_rf.best_score_ else gs_rf
best_est = best_gs.best_estimator_

# ---- fit on train, calibrate on val
# note: CalibratedClassifierCV wraps the whole pipeline (which outputs predict_proba)
cal = CalibratedClassifierCV(best_est, method="isotonic", cv="prefit")
cal.fit(X_val, y_val)

# ---- evaluate on held-out test
probs = cal.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, probs)
aupr = average_precision_score(y_test, probs)
brier = brier_score_loss(y_test, probs)

fpr, tpr, _ = roc_curve(y_test, probs)
prec, rec, _ = precision_recall_curve(y_test, probs)

plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc:.3f})")
plt.tight_layout(); plt.savefig(ART/"roc.png"); plt.close()

plt.figure(); plt.plot(rec,prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUPRC={aupr:.3f})")
plt.tight_layout(); plt.savefig(ART/"pr.png"); plt.close()

frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
plt.figure(); plt.plot(mean_pred, frac_pos, marker="o"); plt.plot([0,1],[0,1],"--")
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.title(f"Reliability (Brier={brier:.3f})")
plt.tight_layout(); plt.savefig(ART/"reliability.png"); plt.close()

# ---- fairness slices (same as before)
slices = pd.DataFrame(index=X_test.index)
slices["sex"] = X.loc[X_test.index, "sex"].astype(int)
ages = X.loc[X_test.index, "age"].astype(float)
slices["age_bucket"] = pd.cut(ages, bins=[0,45,60,120], labels=["lt45","45to60","gt60"])
slices["cp"] = X.loc[X_test.index, "cp"].astype(int)

def auc_by(col):
    out = {}
    for k, idx in slices.groupby(col).groups.items():
        idx = list(idx)
        if len(idx) >= 15 and y_test.loc[idx].nunique() == 2:
            out[str(k)] = float(roc_auc_score(y_test.loc[idx], probs[idx]))
    return out

slice_report = {
    "auc_by_sex": auc_by("sex"),
    "auc_by_age_bucket": auc_by("age_bucket"),
    "auc_by_cp": auc_by("cp")
}

# ---- save artifacts (same names used by the Streamlit app)
joblib.dump({"model": cal, "features": list(X.columns), "preprocessor": pre}, ART/"model.pkl")
with open(ART/"metrics.json","w") as f:
    json.dump({"auc":auc, "aupr":aupr, "brier":brier, 
               "cv_choice":"LR" if best_gs is gs_lr else "RF",
               "cv_best_score": float(best_gs.best_score_),
               **slice_report}, f, indent=2)

print("Saved: artifacts/model.pkl, metrics.json, roc.png, pr.png, reliability.png")