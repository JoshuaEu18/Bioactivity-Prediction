# --- train_gnn_cls.py ---
import os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, RocCurveDisplay
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from rdkit import Chem
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utils import mol_to_graph, scaffold_split, set_seed  # ✅ reproducibility

# --- Reproducibility ---
SEED = 42
g = set_seed(SEED)   # fixes RNG across numpy, torch, python and returns DataLoader generator

# --- Paths ---
data_dir, results_dir = "data/processed", "results"
models_dir, metrics_dir, plots_dir = [os.path.join(results_dir, d) for d in ["models","metrics","plots"]]
for d in [results_dir, models_dir, metrics_dir, plots_dir]:
    os.makedirs(d, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(os.path.join(data_dir, "processed.csv"))
df["active"] = (df["pIC50"] >= 6.0).astype(int)
y_cls = df["active"].values

mols = [Chem.MolFromSmiles(s) for s in df["clean_smiles"]]
graphs = [mol_to_graph(m, yv) for m, yv in zip(mols, y_cls) if m]

# --- Single scaffold split ---
train_df, test_df = scaffold_split(df, frac_train=0.8, seed=SEED)
train_graphs, test_graphs = [graphs[i] for i in train_df.index], [graphs[i] for i in test_df.index]
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, generator=g)
test_loader  = DataLoader(test_graphs, batch_size=64, generator=g)

# --- GNN Classifier ---
class GNNClassifier(nn.Module):
    def __init__(self, in_feats=7, hidden=128, dropout=0.2):
        super().__init__()
        self.conv1, self.bn1 = GCNConv(in_feats, hidden), BatchNorm(hidden)
        self.conv2, self.bn2 = GCNConv(hidden, hidden), BatchNorm(hidden)
        self.lin1, self.lin2 = nn.Linear(hidden, hidden // 2), nn.Linear(hidden // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.lin2(x)).squeeze(-1)

# --- Training setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = GNNClassifier().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

EPOCHS, patience, best_loss, patience_counter = 200, 15, float("inf"), 0
train_losses, val_losses = [], []

print("🧠 Training GNN Classifier...")
for ep in range(EPOCHS):
    # Training
    model.train(); total = 0
    for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS} (Cls)"):
        batch = batch.to(device)
        opt.zero_grad()
        prob = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "edge_attr", None))
        loss = loss_fn(prob, batch.y.float())
        loss.backward(); opt.step()
        total += loss.item()
    avg_train = total / len(train_loader)

    # Validation
    model.eval(); val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            prob = model(batch.x, batch.edge_index, batch.batch)
            val_loss += loss_fn(prob, batch.y.float()).item()
    avg_val = val_loss / len(test_loader)

    train_losses.append(avg_train); val_losses.append(avg_val)
    print(f"Epoch {ep+1} | train {avg_train:.4f} | val {avg_val:.4f}")

    if avg_val < best_loss:
        best_loss = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(models_dir, "best_gnn_cls.pt"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping triggered (Cls)")
            break

# --- Load best model ---
model.load_state_dict(torch.load(os.path.join(models_dir, "best_gnn_cls.pt"), map_location=device))
model.eval()

# --- Evaluate ---
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        prob = model(batch.x, batch.edge_index, batch.batch)
        pred = (prob > 0.5).int()
        y_true += batch.y.cpu().numpy().tolist()
        y_prob += prob.cpu().numpy().tolist()
        y_pred += pred.cpu().numpy().tolist()

auc = float(roc_auc_score(y_true, y_prob))
acc = float(accuracy_score(y_true, y_pred))
f1  = float(f1_score(y_true, y_pred))
metrics = {"GNN_Cls": {"ROC-AUC": auc, "Accuracy": acc, "F1": f1}}

# --- Save metrics ---
pd.DataFrame(metrics).T.to_csv(os.path.join(metrics_dir, "train_gnn_classification_metrics.csv"))
print("📊 Test Results:", metrics)

# --- Save training curve ---
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("GNN Classification Training Curve")
plt.savefig(os.path.join(plots_dir, "train_gnn_cls_training_curve.png"))
plt.close()

# --- Save ROC curve ---
plt.figure()
RocCurveDisplay.from_predictions(y_true, y_prob)
plt.title("ROC Curve - GNN Classifier")
plt.savefig(os.path.join(plots_dir, "train_gnn_cls_roc.png"))
plt.close()

print("✅ GNN classification training finished. Model, metrics, and plots saved.")
