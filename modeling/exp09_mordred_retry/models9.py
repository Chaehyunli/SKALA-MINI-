"""
9차 모델 정의: Mordred 재도전
- desconly|Algo  : Fingerprint 없이 물리·화학 descriptor만 (Descriptor 3 + RDKit 12 + RDKit 217 정리본 168 + Mordred 정리본 700)
                   → 1~3차 모델과 다른 관점이라 앙상블 재료로 기대
- mordredtuned|LightGBM : base 1,042 + Mordred 700에 3차 튜닝 설정 적용 (2차에서는 기본 설정)
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb

HERE = Path(__file__).resolve().parent
M = HERE.parent
for p in ("exp03_tuning", "exp07_seed_avg"):
    sys.path.insert(0, str(M / p))
from load_features import variant_columns   # noqa: E402
from models7 import tuned                   # noqa: E402

NAMES = ["desconly|LightGBM", "desconly|XGBoost", "mordredtuned|LightGBM"]


def desc_cols():
    spec = json.load(open(M / "exp02_feature_variants" / "results" / "block_spec.json"))
    base = variant_columns()["columns"]["base"]
    nonfp = [c for c in base if not c.startswith(("ecfp_", "fcfp_"))]
    return nonfp + spec["rdkit217"]["columns"] + spec["mordred"]["columns"]


def build(name):
    kind, algo = name.split("|")
    if kind == "desconly":
        if algo == "LightGBM":
            mk = lambda: lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=20,
                                            colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                                            verbose=-1, random_state=0, n_jobs=2)
        else:
            mk = lambda: xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=6, colsample_bytree=0.3,
                                           subsample=0.8, tree_method="hist", eval_metric="logloss", random_state=0, n_jobs=2)
        return mk, desc_cols()
    return (lambda: tuned(algo)), variant_columns()["columns"]["+mordred"]
