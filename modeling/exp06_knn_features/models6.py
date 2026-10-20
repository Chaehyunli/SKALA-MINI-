"""6차 모델 정의: 이웃 라벨 특성을 붙인 모델 (이름 → (모델 생성 함수, 입력 컬럼))"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
M = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(M / "exp03_tuning"))
from knnfeat import KnnAugmented          # noqa: E402
from load_features import variant_columns  # noqa: E402

FULLFP = [f"fullecfp_{i}" for i in range(1024)] + [f"fullfcfp_{i}" for i in range(1024)]
CONT15 = ["MolWt", "clogp", "sa_score", "TPSA", "HBD", "HBA", "n_rotb", "n_ring", "n_arom_ring", "frac_csp3",
          "n_halogen", "n_S", "abs_charge", "is_salt", "has_stereo_tag"]
NAMES = ["knn|LightGBM", "knn|XGBoost", "knn|RandomForest", "knnonly|LightGBM", "knnonly|LogReg"]


def _tuned(algo):
    p = json.load(open(M / "exp03_tuning" / "results" / "best_params.json"))["params"][algo]
    if algo == "LightGBM":
        return lgb.LGBMClassifier(**p, subsample_freq=1, verbose=-1, random_state=0, n_jobs=2)
    if algo == "XGBoost":
        return xgb.XGBClassifier(**p, tree_method="hist", eval_metric="logloss", random_state=0, n_jobs=2)
    return RandomForestClassifier(**p, n_jobs=2, random_state=0)


def build(name):
    """name: 'knn|LightGBM' 등 (접두어 '6차:' 없이)"""
    kind, algo = name.split("|")
    if kind == "knn":           # 3차 튜닝 모델 + 이웃 라벨 특성 16개
        return (lambda: KnnAugmented(_tuned(algo))), variant_columns()["columns"]["+maccs"] + FULLFP
    if algo == "LightGBM":      # 이웃 라벨 특성 16개 + 연속형 15개만 쓰는 작은 모델 (앙상블용으로 다른 관점)
        est = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0, n_jobs=2)
    else:
        est = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    return (lambda: KnnAugmented(est)), CONT15 + FULLFP
