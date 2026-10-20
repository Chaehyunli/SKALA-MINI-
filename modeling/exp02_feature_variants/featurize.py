"""
2차 실험: SMILES에서 추가 특성 블록을 계산한다. (train / predict 공통)

블록
- rdkit217 : RDKit 전체 descriptor 217개 (Descriptors.CalcMolDescriptors)
- mordred  : Mordred 2D descriptor 1,613개 (mordredcommunity)
- ptfp     : 제공된 PTFP 1,024비트 (계산 아님, 입력 파일에서 가져옴)
- maccs    : MACCS keys 167비트
- ap_tt    : Atom-pair count 2,048 + Topological torsion count 2,048
- ecfp_cnt : 입체 정보 반영 Morgan radius 2 count 2,048 (ECFP4, chirality on)
- fcfp_cnt : Morgan feature radius 2 count 2,048 (FCFP4)

라벨을 쓰지 않는 정리 규칙 (filter_block)
- 연속형: 결측 20% 초과 컬럼 제거 → 상수 제거 → 기존 연속형 + 앞쪽 컬럼과 |상관| 0.95 이상이면 제거
- Fingerprint형: 0이 아닌 비율이 2~98%인 컬럼만 → 값이 완전히 같은 컬럼 제거
- 정리 기준(남길 컬럼, 결측 대체값)은 개발 세트에서만 계산해 저장하고, 예측 데이터에는 그대로 적용한다
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as G

RDLogger.DisableLog("rdApp.*")

CONT_BLOCKS = {"rdkit217", "mordred"}
FP_BLOCKS = {"ptfp", "maccs", "ap_tt", "ecfp_cnt", "fcfp_cnt"}

_ecfp = G.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
_fcfp = G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen())
_ap = G.GetAtomPairGenerator(fpSize=2048)
_tt = G.GetTopologicalTorsionGenerator(fpSize=2048)


def _mols(smiles):
    ms = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in ms):
        raise ValueError("SMILES 파싱 실패")
    return ms


def rdkit217(smiles):
    rows = [Descriptors.CalcMolDescriptors(m) for m in _mols(smiles)]
    d = pd.DataFrame(rows).astype(float)
    d.columns = [f"rd_{c}" for c in d.columns]
    return d.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)


def mordred_desc(smiles, nproc=2):
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    d = calc.pandas(_mols(smiles), nproc=nproc, quiet=True)
    d = d.apply(pd.to_numeric, errors="coerce").astype(float)
    d.columns = [f"md_{c}" for c in d.columns]
    return d.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)


def _fp(gen, ms, prefix, count=True):
    f = gen.GetCountFingerprintAsNumPy if count else gen.GetFingerprintAsNumPy
    arr = np.vstack([f(m) for m in ms]).astype(np.float32)
    return pd.DataFrame(arr, columns=[f"{prefix}_{i}" for i in range(arr.shape[1])])


def fingerprints(smiles):
    ms = _mols(smiles)
    maccs = np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in ms], dtype=np.float32)
    return {
        "maccs": pd.DataFrame(maccs, columns=[f"maccs_{i}" for i in range(maccs.shape[1])]),
        "ap_tt": pd.concat([_fp(_ap, ms, "ap"), _fp(_tt, ms, "tt")], axis=1),
        "ecfp_cnt": _fp(_ecfp, ms, "ecfpc"),
        "fcfp_cnt": _fp(_fcfp, ms, "fcfpc"),
    }


def filter_block(block, kind, ref_cont=None, corr_thr=0.95, nan_thr=0.2, lo=0.02):
    """라벨 없이 남길 컬럼과 결측 대체값을 정한다. block은 개발 세트 행만 넣는다."""
    if kind == "fp":
        nz = (block != 0).mean()
        keep = nz[(nz >= lo) & (nz <= 1 - lo)].index
        b = block[keep]
        keep = list(b.T.drop_duplicates().index)
        return keep, {}
    b = block.loc[:, block.isna().mean() <= nan_thr]
    med = b.median()
    b = b.fillna(med)
    b = b.loc[:, b.std() > 1e-12]
    base = ref_cont.fillna(ref_cont.median()) if ref_cont is not None else pd.DataFrame(index=b.index)
    allc = pd.concat([base, b], axis=1)
    r = np.abs(np.corrcoef(allc.values.T))
    nb = base.shape[1]
    kept_idx = list(range(nb))
    for j in range(nb, allc.shape[1]):
        if np.all(r[j, kept_idx] < corr_thr):
            kept_idx.append(j)
    keep = [allc.columns[j] for j in kept_idx[nb:]]
    return keep, med[keep].to_dict()


def apply_filter(block, keep, med):
    b = block.reindex(columns=keep)
    return b.fillna(med) if med else b


def read_cache(cache_dir, name):
    """cache/{name}.parquet 또는 용량 때문에 나눠 저장한 cache/{name}_part*.parquet 를 읽는다"""
    from pathlib import Path
    cache_dir = Path(cache_dir)
    one = cache_dir / f"{name}.parquet"
    if one.exists():
        return pd.read_parquet(one)
    parts = sorted(cache_dir.glob(f"{name}_part*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{name} cache 없음 — build_features.py 실행")
    return pd.concat([pd.read_parquet(p) for p in parts], axis=1)
