import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 6차 실험: 이웃 라벨 특성 (kNN label features)

| 항목 | 내용 |
|---|---|
| 목표 | 4차 최고점(스태킹 33개 + 임계값, CV F1 0.8282)을 넘는다 |
| 아이디어 | 예측 데이터는 학습 데이터와 구조가 비슷한 화합물이 많다(유사도 0.8 이상 53%). 가장 비슷한 학습 화합물들의 라벨을 직접 특성으로 주면 도움이 될 수 있다 |
| 특성 | ECFP·FCFP 각각: 최대 유사도, 상위 5개 평균 유사도, 유사도 가중 라벨 평균(k = 1, 3, 5, 10), 유사도 0.7 이상 이웃의 라벨 평균과 개수 → 16개 |
| 누수 방지 | 이웃은 항상 그 모델의 학습 데이터 안에서만 찾는다. 학습 행은 자기 자신을 뺀 이웃(leave-one-out), 검증·테스트·예측 행은 학습 데이터 전체가 이웃 후보 (`knnfeat.KnnAugmented`) |
| 모델 | ① 3차 튜닝 모델 3개(+MACCS 입력) + 이웃 특성 ② 이웃 특성 16개 + 연속형 15개만 쓰는 작은 모델 2개 (LightGBM, 로지스틱 회귀) — 앙상블에서 다른 관점 |
| 비교 | 단독: 3차 같은 모델과 / 앙상블: 4차 최종 스태킹 vs 33개 + 6차 모델 (같은 10 fold, fold 안 임계값, 짝지은 보정 t-검정) |

실행 약 20분. `RUN_CV = False`면 저장 결과를 불러온다.
""")
code("""
import sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
HERE = Path.cwd(); M = HERE.parent
for p in ("common", "exp03_tuning"): sys.path.insert(0, str(M / p))
sys.path.insert(0, str(HERE))
from preprocess import preprocess_train
from evaluate import load_splits, run_cv, corrected_paired_ttest
from load_features import load_all
from knnfeat import full_fp
import models6, stacking as S

warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
RES = HERE / "results"; RES.mkdir(exist_ok=True)
RUN_CV = True
""")
md("## 1. 데이터")
code("""
df, X, y, groups = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
ALL = pd.concat([load_all(df, X), full_fp(df).reset_index(drop=True)], axis=1)
dev = (df.set == "dev").values
Xd = ALL[dev].reset_index(drop=True); yd = y[dev]; smi = df.loc[dev, "SMILES"].values
folds = df.loc[dev, ["fold_r0", "fold_r1"]].reset_index(drop=True)
print("개발 세트:", Xd.shape)
""")
md("## 2. 6차 모델 10-fold 교차검증")
code("""
if RUN_CV:
    sc, oo = [], {}
    for name in models6.NAMES:
        mk, use = models6.build(name); t0 = time.time()
        s, o = run_cv(mk, Xd[use], yd, folds, name)
        sc.append(s)
        for r in o.columns: oo[f"{name}|{r}"] = o[r].values
        print(f"{name:18s} F1 {s.f1.mean():.4f} ± {s.f1.std(ddof=0):.4f} | AUC {s.auc.mean():.4f} | {time.time()-t0:.0f}s", flush=True)
    scores = pd.concat(sc, ignore_index=True)
    oof = pd.DataFrame(oo); oof.insert(0, "SMILES", smi); oof.insert(1, "label", yd)
    scores.to_csv(RES / "cv_scores.csv", index=False); oof.to_csv(RES / "oof_predictions.csv", index=False)
else:
    scores = pd.read_csv(RES / "cv_scores.csv"); oof = pd.read_csv(RES / "oof_predictions.csv")
""")
md("## 3. 단독 비교: 이웃 특성을 붙이기 전(3차 튜닝 모델)과")
code("""
s3 = pd.read_csv(M / "exp03_tuning/results/cv_scores.csv")
rows = []
for algo in ("LightGBM", "XGBoost", "RandomForest"):
    a = s3[s3.model == f"tuned|{algo}"].sort_values(["repeat", "fold"]); b = scores[scores.model == f"knn|{algo}"].sort_values(["repeat", "fold"])
    d = b.f1.values - a.f1.values
    rows.append((algo, a.f1.mean(), b.f1.mean(), d.mean(), S.paired_p(d), int((d > 0).sum()), a.auc.mean(), b.auc.mean()))
single = pd.DataFrame(rows, columns=["모델", "3차 F1", "+이웃 특성 F1", "차이", "p값", "10 fold 중 개선", "3차 AUC", "+이웃 AUC"]).set_index("모델")
display(single.round(4))
display(scores.groupby("model")[["f1", "auc", "precision", "recall"]].mean().sort_values("f1", ascending=False).round(4))
""")
md("## 4. 앙상블: 4차 최종 vs 4차 + 6차 모델")
code("""
pool = S.base_pool(smi); base_names = list(pool)
pool.update(S.read_oof(RES / "oof_predictions.csv", "6차:", smi))
new = [f"6차:{n}" for n in models6.NAMES]
tuned = [k for k in base_names if k.startswith("3차:")]
extra = {"스태킹: 3차 튜닝 3개 + 6차 5개": S.m_stack(pool, tuned + new, yd, C=1.0),
         "스태킹 33개 + 6차 작은 모델 2개": S.m_stack(pool, base_names + new[3:], yd, C=0.1)}
tbl, res = S.compare_with_s4(pool, base_names, new, yd, folds, extra)
tbl.to_csv(RES / "ensemble_compare.csv", index=False)
display(tbl.round(4))

P = pd.DataFrame({k.replace("3차:", "S3 ").replace("6차:", "S6 "): pool[k]["r0"] for k in tuned + new})
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(P.corr(method="spearman"), annot=True, fmt=".2f", cmap="viridis", vmin=0.6, vmax=1, ax=ax[0], annot_kws={"size": 8})
ax[0].set_title("Spearman correlation of OOF predictions")
long = pd.concat([r.assign(method=n) for n, r in res.items()])
long["method"] = long.method.map({"4차 최종: 스태킹 33개": "S4 stack33", "스태킹 33개 + 새 모델": "stack33 + S6 all",
                                  "스태킹: 3차 튜닝 3개 + 6차 5개": "S3 tuned3 + S6", "스태킹 33개 + 6차 작은 모델 2개": "stack33 + S6 small2"})
sns.boxplot(data=long, x="method", y="f1_thr", ax=ax[1], color="#cfe0f3"); sns.stripplot(data=long, x="method", y="f1_thr", ax=ax[1], color="#2b5d9a", size=4)
ax[1].set_xlabel(""); ax[1].set_ylabel("F1 (threshold chosen inside folds)"); ax[1].set_title("Ensemble with kNN-label models"); ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(RES / "fig_knn.png", dpi=130); plt.show()
""")
md("""
## 5. 6차 최종 방법 확정 (제출 파일에 사용)
- 6차 모델이 들어간 방법 중 F1(fold 안 임계값)이 가장 높은 것을 6차 최종으로 한다.
""")
code("""
cand = tbl[tbl["방법"] != "4차 최종: 스태킹 33개"].sort_values("F1(fold 안 임계값)", ascending=False).iloc[0]
METHOD = cand["방법"]
names = {"스태킹 33개 + 새 모델": base_names + new, "스태킹: 3차 튜닝 3개 + 6차 5개": tuned + new,
         "스태킹 33개 + 6차 작은 모델 2개": base_names + new[3:]}[METHOD]
Cc = 1.0 if METHOD.startswith("스태킹: 3차") else 0.1
final = S.finalize(pool, names, yd, Cc, RES / "final_method.json", METHOD, cand["F1(fold 안 임계값)"])
print(METHOD, "| 모델 수", len(names), "| 임계값", final["threshold"], "| CV F1", round(final["cv_f1"], 4),
      "| 4차 최종 대비", round(cand["4차 최종 대비"], 4), "(p=", round(cand["p값"], 3), ")")
""")
md("## 6. 결론")
r_ = open("README.md", encoding="utf-8").read()
md(r_[r_.index("## 판단"):r_.index("## 파일")].replace("\n## ", "\n### ").replace("## 판단", "### 판단"))
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "exp06_knn_features.ipynb"); print("cells", len(C))
