"""16:20까지 GNN 교차검증이 끝나지 않을 때의 예비안: 8차 확률과 GNN 확률을 고정 가중치(logit 0.8 : 0.2)로 결합, 임계값 0.5 (검증 없음)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; M = HERE.parent
sys.path.insert(0, str(M / "common"))
from submission import write_submission
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
p8 = pd.read_csv(M / "exp08_pseudo_label/results/predict_proba.csv"); pg = pd.read_csv(HERE / "results_fast/gnn_predict_full.csv")
z = 0.8 * logit(p8.score_nontoxic.values) + 0.2 * logit(pg.gnn_prob.values)
path, sub = write_submission(p8.SMILES, (z >= 0).astype(int), HERE, "5차_thr0.5")
prev = pd.read_csv(M / "exp08_pseudo_label/predict_output_임채현_8차_thr0.5.csv").iloc[:, 1].values
print(path.name, "| 8차_thr0.5와 다른 행", int((sub.iloc[:, 1].values != prev).sum()))
