# --- Clean output ---
import warnings
import random
import numpy as np
import torch
from utils import clean_smiles, to_molar, to_pIC50, scaffold_split


warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.max_colwidth", 120)

# --- Fix seeds for reproducibility ---
SEED = 42

# Python & NumPy
random.seed(SEED)
np.random.seed(SEED)

# PyTorch (CPU & GPU/MPS)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Ensure deterministic behavior (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# 🔹 Use a seeded generator for DataLoader shuffling
g = torch.Generator().manual_seed(SEED)

# --- Parameters ---
import os

target   = "CHEMBL203"   # EGFR
endpoint = "IC50"
outdir   = "data/processed"

# Create the output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

# Fingerprint parameters
FP_BITS   = 1024
FP_RADIUS = 2

# --- Cleaning and transformation ---
import os
import shutil
import pandas as pd

import argparse
import pandas as pd
from utils import clean_smiles, to_molar, to_pIC50

# --- Arguments ---
parser = argparse.ArgumentParser(description="Preprocess SMILES data")
parser.add_argument("--path_to_smiles_data", required=True, help="Path to input raw SMILES data (CSV)")
parser.add_argument("--path_to_processed_data", required=True, help="Path to save processed dataset")
parser.add_argument("--type_of_processing", choices=["fingerprints", "descriptors", "both"], default="fingerprints")
args = parser.parse_args()

# --- Load input data ---
print(f"📥 Reading input data from {args.path_to_smiles_data}")
df = pd.read_csv(args.path_to_smiles_data)


# --- Cleaning & transformation ---
print("🧹 Cleaning and transforming...")

# 1. keep only valid activity relations
df = df[df["standard_relation"].isin(["=", "~", "≈", None])]

# 2. clean SMILES
df["clean_smiles"] = df["canonical_smiles"].apply(clean_smiles)

# 3. convert activity values to molar
df["molar"] = df.apply(lambda r: to_molar(r["standard_value"], r["standard_units"]), axis=1)

# 4. compute pIC50
df["pIC50"] = df["molar"].apply(to_pIC50)

# 5. drop rows without valid SMILES or pIC50
df = df.dropna(subset=["clean_smiles", "pIC50"]).reset_index(drop=True)

# 6. aggregate duplicates
df = df.groupby(["molecule_chembl_id", "clean_smiles"], as_index=False)["pIC50"].median()

print("✅ After cleaning:", df.shape)
print(df.head())

# --- Save processed file ---
os.makedirs(os.path.dirname(args.path_to_processed_data), exist_ok=True)
print(f"💾 Saving cleaned dataset to {args.path_to_processed_data}")
df.to_csv(args.path_to_processed_data, index=False)

# Save processed file
processed_path = os.path.join(outdir, "processed.csv")
df.to_csv(processed_path, index=False)
print(f"✅ Saved cleaned dataset to {processed_path}")

# --- Also copy to raw folder ---
raw_dir = os.path.join(outdir, "raw")
os.makedirs(raw_dir, exist_ok=True)

raw_path = os.path.join(raw_dir, "cleaned_chembl_data.csv")
shutil.copy(processed_path, raw_path)

print(f"📦 Archived cleaned file to {raw_path}")

# --- Featurisation (Fingerprints + Descriptors) ---
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.impute import SimpleImputer

print("🔬 Featurizing molecules...")

# Load the cleaned dataset
cleaned_path = os.path.join(outdir, "processed.csv")
df = pd.read_csv(cleaned_path)

# Convert SMILES to RDKit molecules
mols = [Chem.MolFromSmiles(s) for s in df["clean_smiles"]]

# Morgan Fingerprints
gen = GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

def fp_array(m): 
    arr = np.zeros((FP_BITS,), dtype=int)
    if m:
        Chem.DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
    return arr

fps = np.array([fp_array(m) for m in mols], dtype=int)

# RDKit Descriptors
desc_funcs = [f for _, f in Descriptors._descList]

def desc_array(m):
    return np.array([f(m) for f in desc_funcs], dtype=float) if m else np.zeros((len(desc_funcs),))

desc = np.array([desc_array(m) for m in mols], dtype=float)

# Combine features: fingerprints + descriptors
X = np.hstack([fps, desc])

# Target values
y = df["pIC50"].values.astype(float)

# Handle any NaN descriptor values
imputer = SimpleImputer(strategy="constant", fill_value=0.0)
X = imputer.fit_transform(X)

print("Feature matrix:", X.shape)

import numpy as np

np.save(os.path.join(outdir, "X.npy"), X)
np.save(os.path.join(outdir, "y.npy"), y)
print("💾 Saved features to X.npy and y.npy")

