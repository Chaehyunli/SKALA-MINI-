"""마감 대응: GNN을 8,000행 전체로 학습해 predict_input 927행 확률을 저장 (5차 제출용)"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; M = HERE.parent
sys.path.insert(0, str(M / "common")); sys.path.insert(0, str(HERE))
from preprocess import preprocess_train, preprocess_predict
from submission import INPUT
import gnn
XD = ["MolWt", "clogp", "sa_score", "TPSA", "HBD", "HBA", "n_rotb", "n_ring", "n_arom_ring", "frac_csp3",
      "n_halogen", "n_S", "abs_charge", "is_salt", "has_stereo_tag"]
df, X, y, g = preprocess_train(M.parent / "train.csv")
pdf, Xp = preprocess_predict(INPUT)
t = time.time()
p = gnn.fit_predict(df["SMILES"].tolist(), y, pdf["SMILES"].tolist(), df[XD].values, pdf[XD].values, epochs=25, seed=0)
pd.DataFrame({"SMILES": pdf["SMILES"], "gnn_prob": p}).to_csv(HERE / "results_fast" / "gnn_predict_full.csv", index=False)
print("done", round(time.time() - t), "s", flush=True)
