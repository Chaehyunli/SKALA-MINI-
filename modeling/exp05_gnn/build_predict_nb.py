import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 5차 제출 파일 만들기: GNN이 들어간 최종 방법으로 predict_input 927행 예측

| 단계 | 내용 |
|---|---|
| 1 | 5차 최종 방법(`results/final_method.json`): 모델 목록(1~3차 모델 + GNN) · 결합 계수 · 임계값 |
| 2 | 테스트 세트 1회 확인: 모든 구성 모델을 개발 6,400행으로 학습 → 테스트 1,600행 예측 → 결합 |
| 3 | 8,000행 전체로 재학습 → predict_input 예측 → `predict_output_임채현_5차.csv` |

실행 시간: 약 35분 (구성 모델 40여 개 × 2회 + GNN × 2회)
""")
code("""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path.cwd(); M = HERE.parent
for p in ("common", "exp03_tuning", "exp06_knn_features"): sys.path.insert(0, str(M / p))
sys.path.insert(0, str(HERE))
from preprocess import preprocess_train, preprocess_predict, load_feature_columns
from evaluate import load_splits, predict_score
from submission import write_submission, test_report, INPUT
from load_features import load_all, load_predict, blocks_for
import registry_ext as R
from knnfeat import full_fp
import gnn

final = json.load(open(HERE / "results" / "final_method.json"))
print(final["method"], "| 모델 수", len(final["models"]), "| 임계값", final["threshold"], "| CV F1", round(final["cv_f1"], 4))
df, X, y, g = preprocess_train(M.parent / "train.csv"); df = load_splits(df); cols = load_feature_columns()
ALL = pd.concat([load_all(df, X), full_fp(df).reset_index(drop=True)], axis=1); dev = (df.set == "dev").values
XD = final["gnn"]["xd_cols"]; EP = final["gnn"]["epochs"]
specs = {n: R.build(n, cols) for n in final["models"] if not n.startswith("5차:")}
need = sorted({c for _, use in specs.values() for c in use if not c.startswith("full")})

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def combine(P):
    if final["kind"] == "mean":
        return np.sum([w * P[n] for n, w in zip(final["models"], final["weights"])], 0)
    z = final["intercept"] + np.sum([w * logit(P[n]) for n, w in zip(final["models"], final["weights"])], 0)
    return 1 / (1 + np.exp(-z))

def predict_all(tr_mask, X_te, df_te):
    P = {}
    for n in final["models"]:
        t0 = time.time()
        if n.startswith("5차:"):
            P[n] = gnn.fit_predict(df.loc[tr_mask, "SMILES"].tolist(), y[tr_mask], df_te["SMILES"].tolist(),
                                   df.loc[tr_mask, XD].values, df_te[XD].values, epochs=EP, seed=0)
        else:
            mk, use = specs[n]; m = mk(); m.fit(ALL.loc[tr_mask, use], y[tr_mask]); P[n] = predict_score(m, X_te[use])
        print(f"{n:35s} {time.time()-t0:5.0f}s", flush=True)
    return P
""")
md("## 테스트 세트 1회 확인")
code("""
P = predict_all(dev, ALL.loc[~dev], df.loc[~dev])
rep = test_report(y[~dev], combine(P), final["threshold"]); rep05 = test_report(y[~dev], combine(P), 0.5)
print("최종 임계값", {k: round(v, 4) for k, v in rep.items()}); print("임계값 0.5 ", {k: round(v, 4) for k, v in rep05.items()})
pd.DataFrame([rep, rep05], index=[final["method"], final["method"] + " (임계값 0.5)"]).to_csv(HERE / "results" / "test_score.csv")
if "5차:gnn" in P:
    print("참고: GNN 단독 테스트", {k: round(v, 4) for k, v in test_report(y[~dev], P["5차:gnn"]).items()})
""")
md("## 8,000행 전체로 재학습 후 예측")
code("""
pdf, Xp = preprocess_predict(INPUT)
XP = pd.concat([load_predict(pdf, Xp, blocks_for(need)), full_fp(pdf).reset_index(drop=True)], axis=1)
P = predict_all(np.ones(len(df), bool), XP, pdf)
p = combine(P)
path, sub = write_submission(pdf["SMILES"], (p >= final["threshold"]).astype(int), HERE, "5차_전체재학습")
print("저장:", path.name, "| 행 수", len(sub), "| 예측 비독성(1) 비율", round(sub.iloc[:, 1].mean(), 3), "(train 0.542)")
pd.DataFrame({"SMILES": pdf["SMILES"], "score_nontoxic": p}).to_csv(HERE / "results" / "predict_proba.csv", index=False)
path05, sub05 = write_submission(pdf["SMILES"], (p >= 0.5).astype(int), HERE, "5차_전체재학습_thr0.5")
print("임계값 0.5 버전:", path05.name, "| 예측 비독성(1) 비율", round(sub05.iloc[:, 1].mean(), 3))
for k, f in (("3차", "exp03_tuning/predict_output_임채현_3차.csv"), ("4차", "exp04_threshold_ensemble/predict_output_임채현_4차.csv"), ("8차_thr0.5", "exp08_pseudo_label/predict_output_임채현_8차_thr0.5.csv")):
    prev = pd.read_csv(M / f).iloc[:, 1].values
    for lab, s_ in (("최종 임계값", sub), ("임계값 0.5", sub05)):
        v = s_.iloc[:, 1].values
        print(f"{k} 제출과 다른 행 ({lab}): {int((prev != v).sum())} (5차만 1: {int(((v == 1) & (prev == 0)).sum())}, {k}만 1: {int(((v == 0) & (prev == 1)).sum())})")
display(sub.head())
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "predict_submission.ipynb"); print("ok")
