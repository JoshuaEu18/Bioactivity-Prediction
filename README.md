# Bioactivity-Prediction
# Bioactivity Prediction with Machine Learning — Dopamine D2 Receptor

1. Project Overview  

This project implements an end-to-end machine learning pipeline to predict the bioactivity of small molecules against the **CHEMBL203 = Epidermal Growth Factor Receptor**.  

The pipeline covers:  
- Data collection from ChEMBL  
- Preprocessing and standardization of bioactivity data  
- Feature extraction (fingerprints + molecular descriptors)  
- Model training (Random Forest, XGBoost, etc.)  
- Evaluation and visualization of results  

## 2. Repository Structure

```text
project-root/
│── data/
│   │── raw/
│   │   │── chembl_data.csv
│   │── processed/
│   │   │── X.npy
│   │   │── y.npy
│   │   │── processed.csv
│
│── src/
│   │── preprocessing.py
│   │── __init__.py
│   │── test_classical_cls.py
│   │── test_classical_reg.py
│   │── test_gnn_reg.py
│   │── test_gnn_cls.py
│   │── train_gnn_cls.py
│   │── train_gnn_reg.py
│   │── train_classical_reg.py
│   │── train_classical_cls.py
│   │── utils.py
│   │── __pycache__/
│   │   │── utils.cpython-310.pyc
│
│── results/
│   │── metrics/
│   │── models/
│   │── plots/
│
│── requirements.txt
│── README.md

3. Setup Instructions 

 Prerequisites
- Python ≥ 3.9  
- Recommended: create a virtual environment (e.g., venv, conda, or poetry)  
- Works on macOS
- name: bioactivity-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - pip:
      - aiohappyeyeballs==2.6.1
      - aiohttp==3.12.15
      - aiosignal==1.4.0
      - alembic==1.16.5
      - anyio==4.10.0
      - appnope==0.1.4
      - argon2-cffi==25.1.0
      - argon2-cffi-bindings==25.1.0
      - arrow==1.3.0
      - asttokens==3.0.0
      - async-lru==2.0.5
      - async-timeout==5.0.1
      - attrs==21.4.0
      - babel==2.17.0
      - beautifulsoup4==4.13.5
      - bleach==6.2.0
      - Brotli==1.1.0
      - cached-property==1.5.2
      - certifi==2025.8.3
      - cffi==1.17.1
      - charset-normalizer==3.4.3
      - chembl-webresource-client==0.10.8
      - colorama==0.4.6
      - colorlog==6.9.0
      - easydict==1.13
      - joblib==1.5.2
      - matplotlib==3.10.6
      - networkx==3.3
      - numpy==1.26.4
      - optuna==4.5.0
      - pandas==2.2.3
      - pillow==11.3.0
      - rdkit==2025.3.6
      - scikit-learn==1.4.2
      - scipy==1.13.1
      - seaborn==0.13.2
      - statsmodels==0.14.5
      - sympy==1.13.3
      - torch==2.8.0
      - torch-geometric==2.6.1
      - torchaudio==2.8.0
      - torchvision==0.23.0
      - tqdm==4.67.1
      - xgboost==2.0.3

Installation

```bash
Clone the repository:
git clone https://github.com/JoshuaEu18/Bioactivity-Prediction.git
cd Bioactivity-Prediction/project-root-1

Create virtual environment:
python -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

4. Usage 

Step 1: Data Preparation

Download ChEMBL data and preprocess SMILES + activity values + Molecular fingerprints + descriptors

```bash
python src/preprocessing.py \
    --path_to_smiles_data data/raw/chembl_data.csv \
    --path_to_processed_data data/processed/processed.csv \
    --type_of_processing both


Step 2: Model training
Training using GNN, RF, XGBoost, SVM, SVC for both regression and classification

example: training GNN classification
```bash
python src/train_gnn_cls.py \
    --input features data/processed/X.npy \
    --input labels data/processed/Y.npy \
    --output results/model/best_gnn_cls.pt \
    --output results/metrics/train_gnn_classification_metrices.csv \

Step 3: Evaluate  
Evaluate performance on the test set

example: Test GNN classification
```bash
python src/test_gnn_cls.py \
    --input results/best_gnn_model.pt \
    --input data/processed/processed.csv \
    --split test
    --output results/metrics/gnn_cls_test_metrics.csv \
    --output results/plots/gnn_cls_test_roc.png \

5. Results 

 Classification Task (Active vs Inactive)

| Model              | Dataset | ROC-AUC | Accuracy | F1    |
| ------------------ | ------- | ------- | -------- | ----- |
| Random Forest      | Train   | 0.8925  | 0.8363   | 0.8870 |
| Random Forest      | Test    | 0.9883  | 0.9592   | 0.9713 |
| Support Vector Clf | Train   | 0.8728  | 0.8243   | 0.8765 |
| Support Vector Clf | Test    | 0.9841  | 0.9546   | 0.9679 |
| XGBoost Clf        | Train   | 0.8970  | 0.8765   | 0.8897 |
| XGBoost Clf        | Test    | 0.9817  | 0.9584   | 0.9706 |
| **GNN Classifier** | Train   | 0.8276  | 0.7920   | 0.8424 |
| **GNN Classifier** | Test    | 0.8276  | 0.7920   | 0.8424 |

---

Regression Task (pIC₅₀ prediction)

| Model              | Dataset | RMSE   | MAE    | R²    |
| ------------------ | ------- | ------ | ------ | ----- |
| Random Forest      | Train   | 0.9017 | 0.6947 | 0.5578 |
| Random Forest      | Test    | 0.5113 | 0.3295 | 0.8602 |
| Support Vector Reg | Train   | 0.9036 | 0.6922 | 0.5539 |
| Support Vector Reg | Test    | 0.4944 | 0.2664 | 0.8693 |
| XGBoost Regr       | Train   | 0.8781 | 0.6741 | 0.5797 |
| XGBoost Regr       | Test    | 0.5254 | 0.3523 | 0.8524 |
| **GNN Regression** | Train   | 1.0414 | 0.8238 | 0.4866 |
| **GNN Regression** | Test    | 1.0414 | 0.8238 | 0.4866 |
