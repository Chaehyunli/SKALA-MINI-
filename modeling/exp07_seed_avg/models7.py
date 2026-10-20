"""
7차 모델 정의 (이름 → (모델 생성 함수, 입력 컬럼))
- seedavg|Algo   : 3차 튜닝 설정을 시드 5개로 학습해 확률 평균 (+MACCS 입력)
- transfer|변형|Algo : 3차 튜닝 설정을 2차에서 앙상블 기여가 컸던 다른 입력에 그대로 적용
"""
import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
M = HERE.parent
sys.path.insert(0, str(M / "exp03_tuning"))
from load_features import variant_columns  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
NAMES = ["seedavg|XGBoost", "seedavg|LightGBM",
         "transfer|+ap_tt|LightGBM", "transfer|+ptfp|LightGBM", "transfer|+mordred|XGBoost", "transfer|count_chiral|RandomForest"]


def tuned(algo, seed=0):
    p = json.load(open(M / "exp03_tuning" / "results" / "best_params.json"))["params"][algo]
    if algo == "LightGBM":
        return lgb.LGBMClassifier(**p, subsample_freq=1, verbose=-1, random_state=seed, n_jobs=2)
    if algo == "XGBoost":
        return xgb.XGBClassifier(**p, tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=2)
    return RandomForestClassifier(**p, n_jobs=2, random_state=seed)


class SeedAvg(BaseEstimator, ClassifierMixin):
    def __init__(self, algo, seeds=SEEDS):
        self.algo = algo; self.seeds = seeds

    def fit(self, X, y):
        self.models_ = [tuned(self.algo, s).fit(X, y) for s in self.seeds]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build(name):
    parts = name.split("|")
    vc = variant_columns()["columns"]
    if parts[0] == "seedavg":
        return (lambda: SeedAvg(parts[1])), vc["+maccs"]
    _, var, algo = parts
    return (lambda: tuned(algo)), vc[var]
