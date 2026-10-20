"""
6차 이후 공통: 모델 이름 → (모델 생성 함수, 입력 컬럼)
- 1~3차 모델: exp04_threshold_ensemble/model_registry.py
- N차 모델(6차~): 그 차수 폴더의 modelsN.py 의 build(이름)
입력 프레임은 load_all 결과 + 이웃 계산용 full Fingerprint(fullecfp_, fullfcfp_) 컬럼을 붙여 쓴다.
"""
import importlib.util
import sys
from pathlib import Path

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M / "exp04_threshold_ensemble"))
import model_registry as R        # noqa: E402

STAGE_DIR = {"6차": ("exp06_knn_features", "models6"), "7차": ("exp07_seed_avg", "models7"),
             "8차": ("exp08_pseudo_label", "models8"), "9차": ("exp09_mordred_retry", "models9")}
_cache = {}


def _mod(stage):
    if stage not in _cache:
        d, m = STAGE_DIR[stage]
        sys.path.insert(0, str(M / d))
        spec = importlib.util.spec_from_file_location(m, M / d / f"{m}.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        _cache[stage] = mod
    return _cache[stage]


def build(name, base_cols):
    stage, rest = name.split(":", 1)
    if stage in ("1차", "2차", "3차"):
        return R.build(name, base_cols)
    return _mod(stage).build(rest)
