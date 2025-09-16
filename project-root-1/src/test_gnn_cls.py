import os, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, RocCurveDisplay
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn as nn, torch.nn.functional as F
from rdkit import Chem

from utils import mol_to_graph, scaffold_split, set_seed

# Paths
DATA_DIR, RESULTS_DIR = "data/processed", "results"
MODELS_DIR, METRICS_DIR, PLOTS_DIR = [os.path.join(RESULTS_DIR,d) for d in ["models","metrics","plots"]]

df = pd.read_csv(os.path.join(DATA_DIR,"processed.csv"))
y_cls = (df["pIC50"] >= 6.0).astype(int).values
mols = [Chem.MolFromSmiles(s) for s in df["clean_smiles"]]
graphs = [mol_to_graph(m,yv) for m,yv in zip(mols,y_cls) if m]
test_df, test_df = scaffold_split(df, frac_train=0.8, seed=42)
test_loader = DataLoader([graphs[i] for i in test_df.index], batch_size=64)

class GNNClassifier(nn.Module):
    def __init__(self,in_feats=7,hidden=128,dropout=0.2):
        super().__init__()
        self.conv1,self.bn1=GCNConv(in_feats,hidden),BatchNorm(hidden)
        self.conv2,self.bn2=GCNConv(hidden,hidden),BatchNorm(hidden)
        self.lin1,self.lin2=nn.Linear(hidden,hidden//2),nn.Linear(hidden//2,1)
        self.dropout=dropout
    def forward(self,x,edge_index,batch,edge_attr=None):
        x=F.relu(self.bn1(self.conv1(x,edge_index)))
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=F.relu(self.bn2(self.conv2(x,edge_index)))
        x=global_mean_pool(x,batch)
        x=F.relu(self.lin1(x))
        x=F.dropout(x,p=self.dropout,training=self.training)
        return torch.sigmoid(self.lin2(x)).squeeze(-1)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=GNNClassifier().to(device)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR,"best_gnn_cls.pt"),map_location=device))
model.eval()

yt, yp_prob, yp_bin=[],[],[]
with torch.no_grad():
    for batch in tqdm(test_loader,desc="Testing GNN Cls"):
        batch=batch.to(device); prob=model(batch.x,batch.edge_index,batch.batch)
        yt+=batch.y.cpu().numpy().tolist(); yp_prob+=prob.cpu().numpy().tolist()
        yp_bin+=(prob>0.5).int().cpu().numpy().tolist()

auc, acc, f1=float(roc_auc_score(yt,yp_prob)), float(accuracy_score(yt,yp_bin)), float(f1_score(yt,yp_bin))
pd.DataFrame([{"Model":"GNN_Classification","ROC-AUC":auc,"Accuracy":acc,"F1":f1}]).to_csv(
    os.path.join(METRICS_DIR,"gnn_cls_test_metrics.csv"), index=False)

plt.figure()
RocCurveDisplay.from_predictions(yt,yp_prob)
plt.title("ROC - GNN Classifier")
plt.savefig(os.path.join(PLOTS_DIR,"gnn_cls_test_roc.png"))
plt.close()

print("✅ GNN Classification test done.")
