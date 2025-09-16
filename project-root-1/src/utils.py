# --- utils.py ---
import os
import math
import random
import numpy as np

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

# ==========================
# Reproducibility
# ==========================
def set_seed(seed: int = 42):
    """Fix random seeds across numpy, torch, and Python for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed)

# ==========================
# Cleaning & Conversion
# ==========================
UNIT_TO_M = {"M": 1, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6,
             "nM": 1e-9, "pM": 1e-12}

def to_molar(val, units):
    """Convert activity values into molar concentration (M)."""
    try:
        return float(val) * UNIT_TO_M.get(units, np.nan)
    except Exception:
        return None

def to_pIC50(molar):
    """Convert molar concentration to pIC50 value."""
    return -math.log10(molar) if molar and molar > 0 else None

def clean_smiles(smi):
    """Standardize and canonicalize a SMILES string."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return None
    
    # Standardize: largest fragment + neutralize
    lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    mol = lfc.choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    
    return Chem.MolToSmiles(mol, canonical=True)

# ==========================
# Scaffold Split
# ==========================
def scaffold_split(df, frac_train=0.8, seed=42):
    """Split dataset into train/test sets by Bemis–Murcko scaffolds."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import random

    scaffolds = {}
    for idx, smi in enumerate(df["clean_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.setdefault(scaffold, []).append(idx)

    sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    rng = random.Random(seed)
    rng.shuffle(sorted_scaffolds)

    n_total = len(df)
    n_train = int(frac_train * n_total)
    train_idx, test_idx = [], []

    for _, idxs in sorted_scaffolds:
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return df.iloc[train_idx], df.iloc[test_idx]
# ==========================
# Mol → Graph (for GNN)
# ==========================
def mol_to_graph(mol, y_val):
    """Convert RDKit Mol to PyTorch Geometric Data object with features."""
    if mol is None:
        return None

    # Atom features
    x = [[
        a.GetAtomicNum(),
        a.GetDegree(),
        a.GetTotalNumHs(),
        a.GetImplicitValence(),
        int(a.GetIsAromatic()),
        a.GetFormalCharge(),
        int(a.GetHybridization())
    ] for a in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    # Bond features
    ei, eb = [[], []], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = b.GetBondTypeAsDouble()
        ei[0] += [i, j]; ei[1] += [j, i]
        eb += [[bond], [bond]]

    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_attr = torch.tensor(eb, dtype=torch.float)
    y_t = torch.tensor([y_val], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_t)

def scaffold_kfold(df, n_splits=5, seed=42):
    """
    Scaffold-based K-Fold cross-validation.
    Yields (train_idx, test_idx) for each fold.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = {}
    for idx, smi in enumerate(df["clean_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.setdefault(scaffold, []).append(idx)

    all_scaffolds = list(scaffolds.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_scaffolds)

    folds = [[] for _ in range(n_splits)]
    for i, group in enumerate(all_scaffolds):
        folds[i % n_splits].extend(group)

    for k in range(n_splits):
        test_idx = folds[k]
        train_idx = [i for j in range(n_splits) if j != k for i in folds[j]]
        yield train_idx, test_idx

# --- Reproducibility helper ---
import torch, random, numpy as np

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Return seeded generator for DataLoader
    return torch.Generator().manual_seed(seed)
