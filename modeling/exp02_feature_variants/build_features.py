"""
정제된 8,000행(train)의 추가 특성 블록을 계산해 cache/에 저장한다.
실행: python build_features.py   (약 10분, Mordred가 대부분)
"""
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from preprocess import preprocess_train            # noqa: E402
import featurize as F                             # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

if __name__ == "__main__":
    df, X, y, g = preprocess_train(HERE.parent.parent / "train.csv")
    smi = df["SMILES"].tolist()
    df[["SMILES"]].to_parquet(CACHE / "index.parquet")
    df[[f"ptfp_{i}" for i in range(1024)]].astype("float32").reset_index(drop=True).to_parquet(CACHE / "ptfp.parquet")

    t = time.time()
    F.rdkit217(smi).to_parquet(CACHE / "rdkit217.parquet"); print("rdkit217", round(time.time() - t), "s", flush=True)
    t = time.time()
    for k, v in F.fingerprints(smi).items():
        v.to_parquet(CACHE / f"{k}.parquet")
    print("fingerprints", round(time.time() - t), "s", flush=True)
    t = time.time()
    F.mordred_desc(smi, nproc=2).to_parquet(CACHE / "mordred.parquet"); print("mordred", round(time.time() - t), "s", flush=True)
