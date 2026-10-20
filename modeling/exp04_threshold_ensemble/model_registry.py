"""
4차 앙상블에 들어간 모델을 이름으로 다시 만든다 (재학습 · 예측용)
- "1차:Name"            → 1차 models.py, 입력 base 1,042컬럼
- "2차:variant|Algo"    → 2차 models2.py, 입력 = 2차 variant 컬럼 (svd는 base + SVD 파이프라인)
- "3차:tuned|Algo"      → 3차 best_params.json, 입력 = 3차 튜닝에 쓴 variant 컬럼
"""
import importlib.util
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M / "exp03_tuning"))
from load_features import variant_columns     # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def build(name, base_cols):
    """반환: (모델 생성 함수, 입력 컬럼 목록)"""
    vc = variant_columns()["columns"]
    fp = [c for c in base_cols if c.startswith(("ecfp_", "fcfp_"))]
    stage, rest = name.split(":", 1)
    if stage == "1차":
        m1 = _load(M / "exp01_model_comparison" / "models.py", "models1")
        cont = [c for c in base_cols if c not in fp and c not in ("is_salt", "has_stereo_tag")]
        return m1.build_models(cont, fp)[rest], list(base_cols)
    if stage == "2차":
        m2 = _load(M / "exp02_feature_variants" / "models2.py", "models2")
        var, algo = rest.split("|")
        if var == "svd":
            return m2.svd_algos(fp, [c for c in base_cols if c not in fp])[algo], list(base_cols)
        return m2.ALGOS[algo], vc[var]
    if stage == "3차":
        import lightgbm as lgb, xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        bp = json.load(open(M / "exp03_tuning" / "results" / "best_params.json"))
        algo = rest.split("|")[1]; p = bp["params"][algo]
        if algo == "LightGBM":
            mk = lambda: lgb.LGBMClassifier(**p, subsample_freq=1, verbose=-1, random_state=0, n_jobs=2)
        elif algo == "XGBoost":
            mk = lambda: xgb.XGBClassifier(**p, tree_method="hist", eval_metric="logloss", random_state=0, n_jobs=2)
        else:
            mk = lambda: RandomForestClassifier(**p, n_jobs=2, random_state=0)
        return mk, vc[bp["variant"]]
    raise ValueError(name)
