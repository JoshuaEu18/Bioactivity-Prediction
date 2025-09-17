try:
    import os
    if "KMP_DUPLICATE_LIB_OK" not in os.environ:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
except ImportError:
    pass

# --- train_classical_cls.py ---
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

from utils import scaffold_kfold  

# -------------------------
# Paths
# -------------------------
data_dir = "data/processed"
results_dir = "results"
models_dir = os.path.join(results_dir, "models")
metrics_dir = os.path.join(results_dir, "metrics")
for d in [results_dir, models_dir, metrics_dir]:
    os.makedirs(d, exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(os.path.join(data_dir, "processed.csv"))
X = np.load(os.path.join(data_dir, "X.npy"))
y_raw = np.load(os.path.join(data_dir, "y.npy"))
y = (y_raw >= 6.0).astype(int)  # binarize for classification


metrics = {}

# -------------------------
# Random Forest
# -------------------------
rf_auc, rf_acc, rf_f1 = [], [], []
for tr, te in scaffold_kfold(df, n_splits=5, seed=42):
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    prob = rf.predict_proba(X[te])[:, 1]
    rf_auc.append(roc_auc_score(y[te], prob))
    rf_acc.append(accuracy_score(y[te], pred))
    rf_f1.append(f1_score(y[te], pred))

joblib.dump(rf, os.path.join(models_dir, "rf_cls.pkl"))
metrics["RandomForest_Cls"] = {"ROC-AUC": np.mean(rf_auc),
                               "Accuracy": np.mean(rf_acc),
                               "F1": np.mean(rf_f1)}

# -------------------------
# SVC
# -------------------------
svc_auc, svc_acc, svc_f1 = [], [], []
for tr, te in scaffold_kfold(df, n_splits=5, seed=42):
    svc = make_pipeline(StandardScaler(), SVC(C=10.0, kernel="rbf", probability=True))
    svc.fit(X[tr], y[tr])
    pred = svc.predict(X[te])
    prob = svc.predict_proba(X[te])[:, 1]
    svc_auc.append(roc_auc_score(y[te], prob))
    svc_acc.append(accuracy_score(y[te], pred))
    svc_f1.append(f1_score(y[te], pred))

joblib.dump(svc, os.path.join(models_dir, "svc_cls.pkl"))
metrics["SVC_Cls"] = {"ROC-AUC": np.mean(svc_auc),
                      "Accuracy": np.mean(svc_acc),
                      "F1": np.mean(svc_f1)}

# -------------------------
# XGBoost
# -------------------------
params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}
xgb_auc, xgb_acc, xgb_f1 = [], [], []
for tr, te in scaffold_kfold(df, n_splits=5, seed=42):
    xgb = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X[tr], y[tr])
    pred = xgb.predict(X[te])
    prob = xgb.predict_proba(X[te])[:, 1]
    xgb_auc.append(roc_auc_score(y[te], prob))
    xgb_acc.append(accuracy_score(y[te], pred))
    xgb_f1.append(f1_score(y[te], pred))

joblib.dump(xgb, os.path.join(models_dir, "xgb_cls.pkl"))
metrics["XGB_Cls"] = {"ROC-AUC": np.mean(xgb_auc),
                      "Accuracy": np.mean(xgb_acc),
                      "F1": np.mean(xgb_f1)}

# -------------------------
# Save all metrics
# -------------------------
pd.DataFrame(metrics).to_csv(os.path.join(metrics_dir, "train_classical_classification_metrics.csv"))
print("✅ Classical classification training done. Models in results/models, metrics in results/metrics.")
