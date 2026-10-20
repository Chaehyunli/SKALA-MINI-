import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 8차 실험: 준지도 학습 (pseudo-labeling)

| 항목 | 내용 |
|---|---|
| 목표 | 지금까지 최고(4·6·7차 최종 중 CV F1 최고)를 넘는다 |
| 아이디어 | 라벨이 없는 predict_input 927행도 구조 정보를 갖고 있다. 모델이 확신하는 행(비독성 확률 ≥ 0.85 또는 ≤ 0.15)에 예측 라벨을 붙여 학습에 추가하면, 채점 데이터와 비슷한 구조를 더 배울 수 있다 |
| 누수 확인 | 927행의 실제 라벨은 쓰지 않는다. CV에서는 각 fold의 학습 부분으로만 의사 라벨을 만들고, 검증 fold로 평가한다 |
| 모델 | 3차 튜닝 LightGBM·XGBoost (+MACCS 입력)에 의사 라벨 추가 |
| 비교 | 단독: 3차 같은 모델과 / 앙상블: 지금까지 최고 스태킹 + 8차 모델 |

실행 약 15분.
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
from evaluate import load_splits, run_cv
from load_features import load_all
import models8, stacking as S

warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
RES = HERE / "results"; RES.mkdir(exist_ok=True)
RUN_CV = True; STAGE = "8차"
""")
code("""
df, X, y, groups = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
ALL = load_all(df, X); dev = (df.set == "dev").values
Xd = ALL[dev].reset_index(drop=True); yd = y[dev]; smi = df.loc[dev, "SMILES"].values
folds = df.loc[dev, ["fold_r0", "fold_r1"]].reset_index(drop=True)
U = models8.unlabeled(); print("라벨 없는 예측 데이터:", U.shape)
""")
md("## 1. 8차 모델 10-fold 교차검증")
code("""
if RUN_CV:
    sc, oo, npl = [], {}, {}
    for name in models8.NAMES:
        mk, use = models8.build(name); t0 = time.time()
        made = []
        def mk2(mk=mk):
            m = mk(); made.append(m); return m
        s, o = run_cv(mk2, Xd[use], yd, folds, name)
        npl[name] = [m.n_pseudo_ for m in made]
        sc.append(s)
        for r in o.columns: oo[f"{name}|{r}"] = o[r].values
        print(f"{name:18s} F1 {s.f1.mean():.4f} ± {s.f1.std(ddof=0):.4f} | AUC {s.auc.mean():.4f} | 의사 라벨 평균 {np.mean(npl[name]):.0f}행 | {time.time()-t0:.0f}s", flush=True)
    scores = pd.concat(sc, ignore_index=True)
    oof = pd.DataFrame(oo); oof.insert(0, "SMILES", smi); oof.insert(1, "label", yd)
    scores.to_csv(RES / "cv_scores.csv", index=False); oof.to_csv(RES / "oof_predictions.csv", index=False)
    json.dump(npl, open(RES / "n_pseudo.json", "w"))
else:
    scores = pd.read_csv(RES / "cv_scores.csv"); oof = pd.read_csv(RES / "oof_predictions.csv")
""")
md("## 2. 단독 비교: 3차 튜닝 모델과")
code("""
s3 = pd.read_csv(M / "exp03_tuning/results/cv_scores.csv")
rows = []
for algo in ("LightGBM", "XGBoost"):
    a = s3[s3.model == f"tuned|{algo}"].sort_values(["repeat", "fold"]); b = scores[scores.model == f"pseudo|{algo}"].sort_values(["repeat", "fold"])
    d = b.f1.values - a.f1.values
    rows.append((algo, a.f1.mean(), b.f1.mean(), d.mean(), S.paired_p(d), int((d > 0).sum()), a.auc.mean(), b.auc.mean()))
display(pd.DataFrame(rows, columns=["모델", "3차 F1", "+의사 라벨 F1", "차이", "p값", "10 fold 중 개선", "3차 AUC", "+의사 라벨 AUC"]).set_index("모델").round(4))
""")
md("## 3. 앙상블: 지금까지 최고 vs + 8차 모델")
code("""
pool = S.full_pool(smi, STAGE); base33 = list(S.base_pool(smi))
label, prior, Cp, prior_f1 = S.best_so_far(STAGE)
pool.update(S.read_oof(RES / "oof_predictions.csv", f"{STAGE}:", smi))
new = [f"{STAGE}:{n}" for n in models8.NAMES]
methods = {"4차 최종: 스태킹 33개": S.m_stack(pool, base33, yd, C=0.1), label: S.m_stack(pool, prior, yd, C=Cp),
           "지금까지 최고 + 8차 2개": S.m_stack(pool, prior + new, yd, C=0.1)}
res = {n: S.evaluate(fn, yd, folds) for n, fn in methods.items()}
ref4 = res["4차 최종: 스태킹 33개"]["f1_thr"].values; refb = res[label]["f1_thr"].values
rows = []
for n, r in res.items():
    d4 = r.f1_thr.values - ref4; db = r.f1_thr.values - refb
    rows.append((n, r.f1.mean(), r.f1_thr.mean(), r.f1_thr.std(ddof=0), r.auc.mean(), d4.mean(), S.paired_p(d4), db.mean(), S.paired_p(db), int((db > 0).sum())))
tbl = pd.DataFrame(rows, columns=["방법", "F1@0.5", "F1(fold 안 임계값)", "F1_std", "AUC", "4차 대비", "p(4차)", "지금까지 최고 대비", "p(최고)", "10 fold 중 개선(최고 대비)"])
tbl.to_csv(RES / "ensemble_compare.csv", index=False); display(tbl.round(4))
""")
md("## 4. 8차 최종 방법 확정")
code("""
cand = tbl[tbl["방법"].str.contains(STAGE)].iloc[0]
final = S.finalize(pool, prior + new, yd, 0.1, RES / "final_method.json", cand["방법"], cand["F1(fold 안 임계값)"])
print(cand["방법"], "| 모델 수", len(final["models"]), "| 임계값", final["threshold"], "| CV F1", round(final["cv_f1"], 4))
""")
md("## 5. 결론")
r_ = open("README.md", encoding="utf-8").read()
md(r_[r_.index("## 판단"):r_.index("## 파일")].replace("\n## ", "\n### ").replace("## 판단", "### 판단"))
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "exp08_pseudo_label.ipynb"); print("cells", len(C))
