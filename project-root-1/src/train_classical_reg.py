# --- train_classical_reg.py ---
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
X, y = np.load(os.path.join(data_dir, "X.npy")), np.load(os.path.join(data_dir, "y.npy"))

metrics = {}

# -------------------------
# Random Forest
# -------------------------
rf_rmse, rf_mae, rf_r2 = [], [], []
for tr, te in scaffold_kfold(df, n_splits=5, seed=42):
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    rf_rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
    rf_mae.append(mean_absolute_error(y[te], pred))
    rf_r2.append(r2_score(y[te], pred))

# Save trained model + metrics
joblib.dump(rf, os.path.join(models_dir, "rf_reg.pkl"))
metrics["RandomForest_Reg"] = {"RMSE": np.mean(rf_rmse),
                               "MAE": np.mean(rf_mae),
                               "R2":  np.mean(rf_r2)}

# -------------------------
# SVR
# -------------------------
svr_rmse, svr_mae, svr_r2 = [], [], []
for tr, te in scaffold_kfold(df, n_splits=5, seed=42):
    svr = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1, kernel="rbf"))
    svr.fit(X[tr], y[tr])
    pred = svr.predict(X[te])
    svr_rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
    svr_mae.append(mean_absolute_error(y[te], pred))
    svr_r2.append(r2_score(y[te], pred))

joblib.dump(svr, os.path.join(models_dir, "svr_reg.pkl"))
metrics["SVR_Reg"] = {"RMSE": np.mean(svr_rmse),
                      "MAE": np.mean(svr_mae),
                      "R2":  np.mean(svr_r2)}

# -------------------------
# XGBoost
# -------------------------
params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}
xgb_rmse, xgb_mae, xgb_r2 = [], [], []
for tr, te in scaffold_kfold(df, n_splits=5, seed=42):
    xgb = XGBRegressor(**params)
    xgb.fit(X[tr], y[tr])
    pred = xgb.predict(X[te])
    xgb_rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
    xgb_mae.append(mean_absolute_error(y[te], pred))
    xgb_r2.append(r2_score(y[te], pred))

joblib.dump(xgb, os.path.join(models_dir, "xgb_reg.pkl"))
metrics["XGB_Reg"] = {"RMSE": np.mean(xgb_rmse),
                      "MAE": np.mean(xgb_mae),
                      "R2":  np.mean(xgb_r2)}

# -------------------------
# Save all metrics
# -------------------------
pd.DataFrame(metrics).to_csv(os.path.join(metrics_dir, "test_classical_regression_metrics.csv"))
print("✅ Classical regression training done. Models in results/models, metrics in results/metrics.")
