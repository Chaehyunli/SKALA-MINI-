import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 1차 제출 파일 만들기: 최고 모델로 predict_input 927행 예측

| 단계 | 내용 |
|---|---|
| 1 | 1차 결과(`results/summary.csv`)에서 F1 1위 모델을 고른다 |
| 2 | 테스트 세트 1회 확인: 개발 6,400행으로 학습 → 테스트 1,600행 평가 (모델 선택에는 쓰지 않은 데이터) |
| 3 | 정제된 8,000행 전체로 다시 학습 |
| 4 | `predict_input.csv`를 같은 `preprocess.py`로 변환 → 예측 → `predict_output_임채현_1차.csv` |
""")
code("""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path.cwd(); sys.path.insert(0, str(HERE.parent / "common"))
from preprocess import preprocess_train, preprocess_predict, load_feature_columns
from evaluate import load_splits
from submission import write_submission, test_report, INPUT
from models import build_models

summary = pd.read_csv(HERE / "results" / "summary.csv", index_col=0)
BEST = summary.index[0]
print("1차 최고 모델:", BEST, "| CV F1", round(summary.loc[BEST, "F1"], 4))

df, X, y, g = preprocess_train(HERE.parent.parent / "train.csv"); cols = load_feature_columns(); df = load_splits(df)
fp_cols = [c for c in cols if c.startswith(("ecfp_", "fcfp_"))]
cont_cols = [c for c in cols if c not in fp_cols and c not in ("is_salt", "has_stereo_tag")]
make = build_models(cont_cols, fp_cols)[BEST]
dev = (df.set == "dev").values
X = X.astype("float32")
""")
md("## 테스트 세트 1회 확인")
code("""
m = make(); m.fit(X[dev], y[dev])
p_test = m.predict_proba(X[~dev])[:, 1]
rep = test_report(y[~dev], p_test)
print({k: round(v, 4) for k, v in rep.items()})
pd.DataFrame([rep], index=[BEST]).to_csv(HERE / "results" / "test_score.csv")
""")
md("## 8,000행 전체로 재학습 후 예측")
code("""
m = make(); m.fit(X, y)
pdf, Xp = preprocess_predict(INPUT)
p = m.predict_proba(Xp.astype("float32"))[:, 1]
path, sub = write_submission(pdf["SMILES"], (p >= 0.5).astype(int), HERE, "1차")
print("저장:", path.name, "| 행 수", len(sub), "| 예측 비독성(1) 비율", round(sub.iloc[:, 1].mean(), 3), "(train 0.542)")
pd.DataFrame({"SMILES": pdf["SMILES"], "prob_nontoxic": p}).to_csv(HERE / "results" / "predict_proba.csv", index=False)
display(sub.head())
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "predict_submission.ipynb"); print("ok")
