"""
학습/검증/테스트 분할을 한 번 만들어 고정한다. (모든 차수 공통)

실행:  python make_splits.py          (modeling/common 폴더에서)
결과:  splits.csv  — SMILES, parent, label, set(dev/test), fold_r0, fold_r1

규칙
- 정제 기준은 mini_project/preprocess.py (라벨 충돌 parent 제외, parent당 1행 → 8,000행)
- 같은 parent가 서로 다른 세트에 들어가지 않도록 parent 기준 그룹 분할 + 라벨 비율 유지
- 테스트 20%: 최종 모델 확정 후 한 번만 평가
- 개발 80%: 5-fold 그룹 CV를 seed 0, 1로 2회 반복 (fold_r0, fold_r1, 테스트 행은 -1)
- SMILES 기준으로 저장하므로 차수별로 전처리를 바꿔도 같은 화합물은 같은 세트·fold에 들어간다
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # mini_project/
sys.path.insert(0, str(ROOT))
from preprocess import preprocess_train      # noqa: E402

TEST_SEED = 42
CV_SEEDS = (0, 1)
N_SPLITS = 5


def make_splits():
    df, _, y, groups = preprocess_train(ROOT / "train.csv")
    out = df[["SMILES", "parent", "label"]].copy()

    # 1) 테스트 20%: 5-fold 그룹 분할 중 한 fold를 테스트로 사용
    sgkf = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=TEST_SEED)
    _, test_idx = next(sgkf.split(out, y, groups))
    out["set"] = "dev"
    out.loc[out.index[test_idx], "set"] = "test"

    # 2) 개발 80%: 5-fold 그룹 CV × 2회
    dev = out["set"] == "dev"
    dev_idx = np.where(dev)[0]
    for r, seed in enumerate(CV_SEEDS):
        col = f"fold_r{r}"
        out[col] = -1
        skf = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
        for k, (_, va) in enumerate(skf.split(dev_idx, y[dev_idx], groups[dev_idx])):
            out.loc[out.index[dev_idx[va]], col] = k

    out.to_csv(HERE / "splits.csv", index=False)
    return out


if __name__ == "__main__":
    s = make_splits()
    print(s["set"].value_counts().to_dict())
    print(s.groupby("set")["label"].mean().round(3).to_dict())
    print("parent 겹침(dev∩test):", len(set(s[s.set == "dev"].parent) & set(s[s.set == "test"].parent)))
