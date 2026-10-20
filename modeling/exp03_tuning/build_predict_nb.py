import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 3차 제출 파일 만들기: 튜닝된 최고 모델로 predict_input 927행 예측

| 단계 | 내용 |
|---|---|
| 1 | 3차 결과(`results/before_after.csv`)에서 튜닝 후 F1이 가장 높은 모델과 그 설정(`best_params.json`) |
| 2 | 테스트 세트 1회 확인: 개발 6,400행으로 학습 → 테스트 1,600행 평가 |
| 3 | 8,000행 전체로 재학습 |
| 4 | predict_input에 2차와 같은 특성을 만들어 예측 → `predict_output_임채현_3차.csv` |
""")
code("""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb, xgboost as xgb
HERE = Path.cwd(); M = HERE.parent
sys.path.insert(0, str(M / "common"))
from preprocess import preprocess_train, preprocess_predict
from evaluate import load_splits
from submission import write_submission, test_report, INPUT
from load_features import load_all, load_predict, blocks_for, variant_columns

SEED, N_JOBS = 0, 2
def make(algo, p):
    if algo == "LightGBM": return lgb.LGBMClassifier(**p, subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=N_JOBS)
    if algo == "XGBoost": return xgb.XGBClassifier(**p, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=N_JOBS)
    return RandomForestClassifier(**p, n_jobs=N_JOBS, random_state=SEED)

ba = pd.read_csv(HERE / "results" / "before_after.csv", index_col=0)
ALGO = ba["튜닝 후 F1"].idxmax()
bp = json.load(open(HERE / "results" / "best_params.json")); VAR = bp["variant"]; P = bp["params"][ALGO]
print("3차 최고:", ALGO, "| 변형", VAR, "| CV F1", round(ba.loc[ALGO, "튜닝 후 F1"], 4)); print(P)

df, X, y, g = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
use = variant_columns()["columns"][VAR]
XA = load_all(df, X)[use]; dev = (df.set == "dev").values
""")
md("## 테스트 세트 1회 확인")
code("""
m = make(ALGO, P); m.fit(XA[dev], y[dev])
rep = test_report(y[~dev], m.predict_proba(XA[~dev])[:, 1])
print({k: round(v, 4) for k, v in rep.items()})
pd.DataFrame([rep], index=[ALGO]).to_csv(HERE / "results" / "test_score.csv")
""")
md("## 8,000행 전체로 재학습 후 예측")
code("""
m = make(ALGO, P); m.fit(XA, y)
pdf, Xp = preprocess_predict(INPUT)
XP = load_predict(pdf, Xp, blocks_for(use))[use]
assert XP.isna().sum().sum() == 0
p = m.predict_proba(XP)[:, 1]
path, sub = write_submission(pdf["SMILES"], (p >= 0.5).astype(int), HERE, "3차")
print("저장:", path.name, "| 행 수", len(sub), "| 예측 비독성(1) 비율", round(sub.iloc[:, 1].mean(), 3), "(train 0.542)")
pd.DataFrame({"SMILES": pdf["SMILES"], "prob_nontoxic": p}).to_csv(HERE / "results" / "predict_proba.csv", index=False)
display(sub.head())
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "predict_submission.ipynb"); print("ok")
