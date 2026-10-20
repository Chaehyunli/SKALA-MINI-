"""
2차에서 만든 특성 블록(cache + 정리 기준)을 불러와 변형별 특성 행렬을 만든다.
- 8,000행 전체(개발 + 테스트)에 2차의 정리 기준(남길 컬럼, 결측 대체값)을 그대로 적용한다.
- 정리 기준은 2차에서 개발 세트로만 계산했으므로 테스트 정보는 들어가지 않는다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP02 = HERE.parent / "exp02_feature_variants"
sys.path.insert(0, str(EXP02))
import featurize as F                     # noqa: E402


def load_all(df_base, X_base):
    """df_base/X_base: 이 폴더 preprocess_train 결과 (8,000행). 반환: 모든 후보 컬럼이 들어간 DataFrame"""
    idx = pd.read_parquet(EXP02 / "cache" / "index.parquet")
    assert (idx["SMILES"].values == df_base["SMILES"].values).all()
    spec = json.load(open(EXP02 / "results" / "block_spec.json"))
    parts = [X_base.reset_index(drop=True)]
    for b, s in spec.items():
        raw = F.read_cache(EXP02 / "cache", b)
        parts.append(F.apply_filter(raw, s["columns"], s["median"]))
    return pd.concat(parts, axis=1).astype("float32")


def load_predict(pdf, Xp, blocks_needed=None):
    """predict_input을 같은 방식으로 변환: 블록을 새로 계산하고 2차 정리 기준(컬럼·결측 대체값)을 그대로 적용"""
    spec = json.load(open(EXP02 / "results" / "block_spec.json"))
    blocks_needed = blocks_needed or list(spec)
    smi = pdf["SMILES"].tolist()
    raw = {}
    if "rdkit217" in blocks_needed:
        raw["rdkit217"] = F.rdkit217(smi)
    if "mordred" in blocks_needed:
        raw["mordred"] = F.mordred_desc(smi, nproc=2)
    if "ptfp" in blocks_needed:
        raw["ptfp"] = pdf[[f"ptfp_{i}" for i in range(1024)]].astype("float32").reset_index(drop=True)
    if set(blocks_needed) & {"maccs", "ap_tt", "ecfp_cnt", "fcfp_cnt"}:
        raw.update(F.fingerprints(smi))
    parts = [Xp.reset_index(drop=True)]
    for b, s in spec.items():
        if b in raw:
            parts.append(F.apply_filter(raw[b], s["columns"], s["median"]))
    return pd.concat(parts, axis=1).astype("float32")


def blocks_for(columns):
    """컬럼 목록에 필요한 블록 이름"""
    spec = json.load(open(EXP02 / "results" / "block_spec.json"))
    cs = set(columns)
    return [b for b, s in spec.items() if cs & set(s["columns"])]


def variant_columns():
    return json.load(open(EXP02 / "results" / "variant_columns.json"))
