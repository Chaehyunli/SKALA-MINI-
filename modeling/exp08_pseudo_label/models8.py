"""
8차 모델 정의: 준지도 학습 (pseudo-labeling)
- 라벨이 없는 predict_input 927행을 활용한다.
  1) 학습 데이터로 모델을 학습 → 927행 예측
  2) 확신이 높은 행(비독성 확률 ≥ HI 또는 ≤ LO)에 예측 라벨을 붙여 학습 데이터에 추가
  3) 늘어난 데이터로 다시 학습
- 927행의 실제 라벨은 쓰지 않으므로 누수가 아니다. CV에서는 fold의 학습 부분만으로 1)을 하고, 검증 fold로 평가한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

HERE = Path(__file__).resolve().parent
M = HERE.parent
for p in ("common", "exp03_tuning", "exp07_seed_avg"):
    sys.path.insert(0, str(M / p))
from load_features import variant_columns  # noqa: E402
from models7 import tuned                  # noqa: E402

NAMES = ["pseudo|LightGBM", "pseudo|XGBoost"]
HI, LO = 0.85, 0.15
_UNL = {}


def unlabeled():
    """predict_input 927행을 학습과 같은 방식으로 변환 (+MACCS 변형 컬럼)"""
    if "X" not in _UNL:
        sys.path.insert(0, str(M / "exp03_tuning"))
        from preprocess import preprocess_predict
        from load_features import load_predict
        from submission import INPUT
        pdf, Xp = preprocess_predict(INPUT)
        _UNL["X"] = load_predict(pdf, Xp, ["maccs"])
    return _UNL["X"]


class PseudoLabel(BaseEstimator, ClassifierMixin):
    def __init__(self, algo, hi=HI, lo=LO):
        self.algo = algo; self.hi = hi; self.lo = lo

    def fit(self, X, y):
        U = unlabeled()[list(X.columns)]
        m0 = tuned(self.algo).fit(X, y)
        p = m0.predict_proba(U)[:, 1]
        sel = (p >= self.hi) | (p <= self.lo)
        self.n_pseudo_ = int(sel.sum())
        X2 = pd.concat([X.reset_index(drop=True), U[sel].reset_index(drop=True)], ignore_index=True)
        y2 = np.concatenate([np.asarray(y), (p[sel] >= 0.5).astype(int)])
        self.model_ = tuned(self.algo).fit(X2, y2)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def build(name):
    algo = name.split("|")[1]
    return (lambda: PseudoLabel(algo)), variant_columns()["columns"]["+maccs"]
