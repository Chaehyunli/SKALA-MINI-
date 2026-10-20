import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 2차 제출 파일 만들기: 최고 (입력 변형 × 모델)로 predict_input 927행 예측

| 단계 | 내용 |
|---|---|
| 1 | 2차 결과(`results/cv_scores.csv`)에서 F1이 가장 높은 (변형, 모델)을 고른다 (base = 1차 결과는 제외) |
| 2 | 테스트 세트 1회 확인: 개발 6,400행으로 학습 → 테스트 1,600행 평가 |
| 3 | 8,000행 전체로 재학습 |
| 4 | predict_input에 같은 특성(추가 블록 계산 + 2차 정리 기준)을 만들어 예측 → `predict_output_임채현_2차.csv` |
""")
code("""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.svm import SVC
HERE = Path.cwd(); M = HERE.parent
sys.path.insert(0, str(M / "common")); sys.path.insert(0, str(M / "exp03_tuning"))
from preprocess import preprocess_train, preprocess_predict, load_feature_columns
from evaluate import load_splits, predict_score
from submission import write_submission, test_report, INPUT
from load_features import load_all, load_predict, blocks_for, variant_columns
from models2 import ALGOS, svd_algos

sc = pd.read_csv(HERE / "results" / "cv_scores.csv")
sc = sc[~sc.model.str.startswith("base|")]
f1 = sc.groupby("model").f1.mean().sort_values(ascending=False)
BEST = f1.index[0]; VAR, ALGO = BEST.split("|")
print("2차 최고:", BEST, "| CV F1", round(f1.iloc[0], 4)); display(f1.head(5).round(4))

df, X, y, g = preprocess_train(M.parent / "train.csv"); df = load_splits(df); cols = load_feature_columns()
ALL = load_all(df, X)
vc = variant_columns()["columns"]
use = vc["base"] if VAR == "svd" else vc[VAR]
fp_base = [c for c in cols if c.startswith(("ecfp_", "fcfp_"))]
make = svd_algos(fp_base, [c for c in cols if c not in fp_base])[ALGO] if VAR == "svd" else ALGOS[ALGO]
XA = ALL[use]; dev = (df.set == "dev").values
print("입력 컬럼 수:", XA.shape[1])
""")
md("## 테스트 세트 1회 확인")
code("""
m = make(); m.fit(XA[dev], y[dev])
rep = test_report(y[~dev], predict_score(m, XA[~dev]))
print({k: round(v, 4) for k, v in rep.items()})
pd.DataFrame([rep], index=[BEST]).to_csv(HERE / "results" / "test_score.csv")
""")
md("## 8,000행 전체로 재학습 후 예측")
code("""
m = make(); m.fit(XA, y)
pdf, Xp = preprocess_predict(INPUT)
XP = load_predict(pdf, Xp, blocks_for(use))[use]
assert XP.isna().sum().sum() == 0 and list(XP.columns) == list(XA.columns)
p = predict_score(m, XP)
path, sub = write_submission(pdf["SMILES"], (p >= 0.5).astype(int), HERE, "2차")
print("저장:", path.name, "| 행 수", len(sub), "| 예측 비독성(1) 비율", round(sub.iloc[:, 1].mean(), 3), "(train 0.542)")
pd.DataFrame({"SMILES": pdf["SMILES"], "score_nontoxic": p}).to_csv(HERE / "results" / "predict_proba.csv", index=False)
display(sub.head())
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "predict_submission.ipynb"); print("ok")
