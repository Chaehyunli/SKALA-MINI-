"""
제출 파일 작성 · 검증과 테스트 세트 1회 평가 (모든 차수 공통)

제출 형식은 교수님 예시 파일(predict_output_example.csv)을 따른다:
  컬럼 SMILES, output / 927행 / predict_input.csv와 같은 행 순서 / 값 0 또는 1 / 줄바꿈 CRLF
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent.parent      # mini_project/
INPUT = ROOT / "predict_input.csv"
EXAMPLE = ROOT / "predict_output_example.csv"
AUTHOR = "임채현"


def write_submission(smiles, pred, out_dir, stage):
    ex = pd.read_csv(EXAMPLE)
    pred = np.asarray(pred).astype(int)
    sub = pd.DataFrame({"SMILES": list(smiles), ex.columns[1]: pred})
    assert len(sub) == len(ex) == 927, f"행 수 {len(sub)}"
    assert (sub["SMILES"].values == ex["SMILES"].values).all(), "SMILES 순서가 예시와 다름"
    assert set(np.unique(pred)) <= {0, 1}, "값은 0/1만"
    assert list(sub.columns) == list(ex.columns), "컬럼명이 예시와 다름"
    path = Path(out_dir) / f"predict_output_{AUTHOR}_{stage}.csv"
    sub.to_csv(path, index=False, lineterminator="\r\n")
    return path, sub


def test_report(y_test, p_test, thr=0.5):
    pred = (p_test >= thr).astype(int)
    return {"F1": f1_score(y_test, pred), "AUC": roc_auc_score(y_test, p_test),
            "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred), "threshold": thr}
