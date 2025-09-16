# --- test_classical_cls_all.py ---
import os, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, RocCurveDisplay

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
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y_cls = (df["pIC50"] >= 6.0).astype(int).values  # active if pIC50 ≥ 6

# -------------------------
# Models to test
# -------------------------
model_files = ["rf_cls.pkl", "svc_cls.pkl", "xgb_cls.pkl"]
all_metrics = []

for model_name in model_files:
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"⚠️ Model {model_name} not found, skipping.")
        continue

    # Load model
    clf = joblib.load(model_path)

    # Predict
    probas = clf.predict_proba(X)[:, 1]
    preds  = clf.predict(X)

    auc = roc_auc_score(y_cls, probas)
    acc = accuracy_score(y_cls, preds)
    f1  = f1_score(y_cls, preds)

    # Print + collect metrics
    print(f"📊 {model_name} | AUC={auc:.3f}, Acc={acc:.3f}, F1={f1:.3f}")
    all_metrics.append({"Model": model_name, "ROC-AUC": auc, "Accuracy": acc, "F1": f1})

    # Save ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_cls, probas)
    plt.title(f"ROC Curve - {model_name}")
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.replace('.pkl','')}_roc.png"))
    plt.close()

# -------------------------
# Save all metrics
# -------------------------
if all_metrics:
    metrics_path = os.path.join(METRICS_DIR, "test_classical_classification_metrics.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"✅ Metrics saved to {metrics_path}")
else:
    print("❌ No models tested, nothing saved.")
