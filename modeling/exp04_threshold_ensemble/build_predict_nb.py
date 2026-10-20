import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 4차 제출 파일 만들기: 최종 앙상블로 predict_input 927행 예측

| 단계 | 내용 |
|---|---|
| 1 | 4차에서 확정한 방법(`results/final_method.json`): 모델 목록 · 가중치(또는 스태킹 계수) · 임계값 |
| 2 | 테스트 세트 1회 확인: 구성 모델을 개발 6,400행으로 학습 → 테스트 1,600행 예측 → 같은 가중치·임계값으로 결합 |
| 3 | 구성 모델을 8,000행 전체로 재학습 |
| 4 | predict_input 특성 생성 → 모델별 예측 → 결합 → `predict_output_임채현_4차.csv` |
""")
code("""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path.cwd(); M = HERE.parent
sys.path.insert(0, str(M / "common")); sys.path.insert(0, str(M / "exp03_tuning")); sys.path.insert(0, str(HERE))
from preprocess import preprocess_train, preprocess_predict, load_feature_columns
from evaluate import load_splits, predict_score
from submission import write_submission, test_report, INPUT
from load_features import load_all, load_predict, blocks_for
import model_registry as R

final = json.load(open(HERE / "results" / "final_method.json"))
print(json.dumps({k: v for k, v in final.items() if k != "weights"}, ensure_ascii=False, indent=1))
df, X, y, g = preprocess_train(M.parent / "train.csv"); df = load_splits(df); cols = load_feature_columns()
ALL = load_all(df, X); dev = (df.set == "dev").values
specs = {n: R.build(n, cols) for n in final["models"]}
need = sorted({c for _, use in specs.values() for c in use})

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def combine(P):
    if final["kind"] == "mean":
        return np.sum([w * P[n] for n, w in zip(final["models"], final["weights"])], 0)
    z = final["intercept"] + np.sum([w * logit(P[n]) for n, w in zip(final["models"], final["weights"])], 0)
    return 1 / (1 + np.exp(-z))
""")
md("## 테스트 세트 1회 확인")
code("""
P = {}
for n, (mk, use) in specs.items():
    m = mk(); m.fit(ALL.loc[dev, use], y[dev]); P[n] = predict_score(m, ALL.loc[~dev, use])
p_test = combine(P)
rep = test_report(y[~dev], p_test, final["threshold"])
print({k: round(v, 4) for k, v in rep.items()})
pd.DataFrame([rep], index=[final["method"]]).to_csv(HERE / "results" / "test_score.csv")
""")
md("## 8,000행 전체로 재학습 후 예측")
code("""
pdf, Xp = preprocess_predict(INPUT)
XP = load_predict(pdf, Xp, blocks_for(need))
P = {}
for n, (mk, use) in specs.items():
    m = mk(); m.fit(ALL[use], y); P[n] = predict_score(m, XP[use])
p = combine(P)
path, sub = write_submission(pdf["SMILES"], (p >= final["threshold"]).astype(int), HERE, "4차")
print("저장:", path.name, "| 행 수", len(sub), "| 예측 비독성(1) 비율", round(sub.iloc[:, 1].mean(), 3), "(train 0.542)")
pd.DataFrame({"SMILES": pdf["SMILES"], "score_nontoxic": p}).to_csv(HERE / "results" / "predict_proba.csv", index=False)
display(sub.head())
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "predict_submission.ipynb"); print("ok")
