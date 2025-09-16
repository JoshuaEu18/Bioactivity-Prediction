# --- test_classical_reg_all.py ---
import os, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------
# Paths
# -------------------------
DATA_DIR, RESULTS_DIR = "data/processed", "results"
MODELS_DIR, METRICS_DIR, PLOTS_DIR = [os.path.join(RESULTS_DIR, d) for d in ["models","metrics","plots"]]
for d in [RESULTS_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "processed.csv"))
X, y = np.load(os.path.join(DATA_DIR, "X.npy")), np.load(os.path.join(DATA_DIR, "y.npy"))

# -------------------------
# Models to test
# -------------------------
model_files = ["rf_reg.pkl", "svr_reg.pkl", "xgb_reg.pkl"]
all_metrics = []

for model_name in model_files:
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"⚠️ Model {model_name} not found, skipping.")
        continue

    # Load model
    model = joblib.load(model_path)

    # Predict
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae  = mean_absolute_error(y, pred)
    r2   = r2_score(y, pred)

    # Print + collect metrics
    print(f"📊 {model_name} | RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
    all_metrics.append({"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2})

    # Save scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel("True pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(f"{model_name} Predictions")
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.replace('.pkl','')}_scatter.png"))
    plt.close()

# -------------------------
# Save all metrics
# -------------------------
if all_metrics:
    metrics_path = os.path.join(METRICS_DIR, "test_classical_regression_metrics.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"✅ Metrics saved to {metrics_path}")
else:
    print("❌ No models tested, nothing saved.")