"""
데이터분석 Mini Project (2조) — 학습/예측 공통 전처리

EDA(EDA_Day1_2조.ipynb)에서 확정한 규칙을 train.csv와 predict_input.csv에 똑같이 적용한다.

사용 예
    from preprocess import preprocess_train, preprocess_predict
    train_df, X, y, groups = preprocess_train("train.csv")
    pred_df, X_pred = preprocess_predict("predict_input.csv")

규칙
1. parent(가장 큰 조각 + 입체정보 제거) 계산
2. (train만) 라벨이 충돌하는 parent 그룹 제외, parent당 1행만 남김
3. RDKit 파생 변수 12개 계산
4. feature_columns.json에 저장된 컬럼만 선택 (fingerprint 필터는 train 기준으로 이미 확정된 목록을 그대로 사용 —
   predict_input에서 다시 계산하지 않음)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
FEATURE_FILE = HERE / "feature_columns.json"

RDKIT_FEATURES = ["TPSA", "HBD", "HBA", "n_rotb", "n_ring", "n_arom_ring", "frac_csp3",
                  "n_halogen", "n_S", "abs_charge", "is_salt", "has_stereo_tag"]


def parent_smiles(mol):
    """가장 큰 조각(염 제거) + 입체정보 제거 canonical SMILES"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    big = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(big, isomericSmiles=False)


def rdkit_features(smiles):
    m = Chem.MolFromSmiles(smiles)
    atoms = [a.GetSymbol() for a in m.GetAtoms()]
    return {
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "HBD": rdMolDescriptors.CalcNumHBD(m),
        "HBA": rdMolDescriptors.CalcNumHBA(m),
        "n_rotb": rdMolDescriptors.CalcNumRotatableBonds(m),
        "n_ring": rdMolDescriptors.CalcNumRings(m),
        "n_arom_ring": rdMolDescriptors.CalcNumAromaticRings(m),
        "frac_csp3": rdMolDescriptors.CalcFractionCSP3(m),
        "n_halogen": sum(a in ("F", "Cl", "Br", "I") for a in atoms),
        "n_S": atoms.count("S"),
        "abs_charge": sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()),
        "is_salt": int(len(Chem.GetMolFrags(m)) > 1),
        "has_stereo_tag": int("@" in smiles),
    }


def _add_meta_and_features(raw):
    out = raw.copy()
    mols = [Chem.MolFromSmiles(s) for s in out["SMILES"]]
    if any(m is None for m in mols):
        bad = out.loc[[m is None for m in mols], "SMILES"].tolist()
        raise ValueError(f"SMILES 파싱 실패: {bad[:5]}")
    out["parent"] = [parent_smiles(m) for m in mols]
    out["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    feats = pd.DataFrame([rdkit_features(s) for s in out["SMILES"]], index=out.index)
    return pd.concat([out, feats], axis=1)


def load_feature_columns(path=FEATURE_FILE):
    """groups(종류별 컬럼 목록)를 order 순서대로 이어 붙여 학습 컬럼 목록을 만든다"""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    return [c for g in spec["order"] for c in spec["groups"][g]]


def preprocess_train(path="train.csv", feature_path=FEATURE_FILE):
    raw = pd.read_csv(path)
    df = _add_meta_and_features(raw)
    nun = df.groupby("parent")["label"].nunique()
    conflict = nun[nun > 1].index
    df = df[~df["parent"].isin(conflict)].drop_duplicates("parent").reset_index(drop=True)
    cols = load_feature_columns(feature_path)
    X = df[cols]
    y = df["label"].astype(int).values
    groups = pd.factorize(df["parent"])[0]
    return df, X, y, groups


def preprocess_predict(path="predict_input.csv", feature_path=FEATURE_FILE):
    raw = pd.read_csv(path)
    df = _add_meta_and_features(raw)          # 예측 데이터는 행을 지우지 않음 (927행 그대로 제출)
    cols = load_feature_columns(feature_path)
    return df, df[cols]
