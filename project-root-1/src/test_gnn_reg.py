import os, numpy as np, torch, pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn as nn, torch.nn.functional as F
from rdkit import Chem

from utils import scaffold_split, mol_to_graph, set_seed

# Paths
DATA_DIR, RESULTS_DIR = "data/processed", "results"
MODELS_DIR, METRICS_DIR = os.path.join(RESULTS_DIR,"models"), os.path.join(RESULTS_DIR,"metrics")

df = pd.read_csv(os.path.join(DATA_DIR,"processed.csv"))
y, mols = df["pIC50"].values.astype(float), [Chem.MolFromSmiles(s) for s in df["clean_smiles"]]
graphs = [mol_to_graph(m,yv) for m,yv in zip(mols,y) if m]
train_df, test_df = scaffold_split(df, frac_train=0.8, seed=42)
test_loader = DataLoader([graphs[i] for i in test_df.index], batch_size=64)

class GNNRegressor(nn.Module):
    def __init__(self,in_feats=7,hidden=128,dropout=0.2):
        super().__init__()
        self.conv1, self.bn1 = GCNConv(in_feats,hidden), BatchNorm(hidden)
        self.conv2, self.bn2 = GCNConv(hidden,hidden), BatchNorm(hidden)
        self.lin1, self.lin2 = nn.Linear(hidden,hidden//2), nn.Linear(hidden//2,1)
        self.dropout=dropout
    def forward(self,x,edge_index,batch,edge_attr=None):
        x=F.relu(self.bn1(self.conv1(x,edge_index)))
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=F.relu(self.bn2(self.conv2(x,edge_index)))
        x=global_mean_pool(x,batch)
        x=F.relu(self.lin1(x))
        x=F.dropout(x,p=self.dropout,training=self.training)
        return self.lin2(x).squeeze(-1)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=GNNRegressor().to(device)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR,"best_gnn_reg.pt"),map_location=device))
model.eval()

yt, yp=[],[]
with torch.no_grad():
    for batch in tqdm(test_loader,desc="Testing GNN Reg"):
        batch=batch.to(device); pred=model(batch.x,batch.edge_index,batch.batch)
        yt+=batch.y.cpu().numpy().tolist(); yp+=pred.cpu().numpy().tolist()

rmse, mae, r2=float(np.sqrt(mean_squared_error(yt,yp))), float(mean_absolute_error(yt,yp)), float(r2_score(yt,yp))
pd.DataFrame([{"Model":"GNN_Regression","RMSE":rmse,"MAE":mae,"R2":r2}]).to_csv(
    os.path.join(METRICS_DIR,"gnn_reg_test_metrics.csv"), index=False)
print("✅ GNN Regression test done.")
