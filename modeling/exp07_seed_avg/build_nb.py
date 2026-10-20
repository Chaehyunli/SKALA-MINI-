import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 7차 실험: 시드 평균과 튜닝 설정 옮기기

| 항목 | 내용 |
|---|---|
| 목표 | 지금까지 최고(4차 또는 6차 최종)를 넘는다 |
| ① 시드 평균 | 3차 튜닝 XGBoost·LightGBM을 시드 5개로 학습해 확률 평균. 3차 단일 모델이 테스트에서 0.809로 흔들렸던 문제(분산)를 줄인다 |
| ② 튜닝 설정 옮기기 | 2차에서 앙상블 기여가 컸던 입력(+AP/TT, +PTFP, +Mordred, 입체 count FP)에 3차 튜닝 설정을 그대로 적용. 2차에서는 기본 설정이었다 |
| 비교 | 같은 10 fold, fold 안 임계값, 짝지은 보정 t-검정. 기준 = 지금까지 최고 스태킹 |

실행 약 30분. `RUN_CV = False`면 저장 결과를 불러온다.
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
import models7, stacking as S

warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
RES = HERE / "results"; RES.mkdir(exist_ok=True)
RUN_CV = True
""")
md("## 1. 데이터")
code("""
df, X, y, groups = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
ALL = load_all(df, X); dev = (df.set == "dev").values
Xd = ALL[dev].reset_index(drop=True); yd = y[dev]; smi = df.loc[dev, "SMILES"].values
folds = df.loc[dev, ["fold_r0", "fold_r1"]].reset_index(drop=True)
""")
md("## 2. 7차 모델 10-fold 교차검증")
code("""
if RUN_CV:
    sc, oo = [], {}
    for name in models7.NAMES:
        mk, use = models7.build(name); t0 = time.time()
        s, o = run_cv(mk, Xd[use], yd, folds, name)
        sc.append(s)
        for r in o.columns: oo[f"{name}|{r}"] = o[r].values
        print(f"{name:34s} F1 {s.f1.mean():.4f} ± {s.f1.std(ddof=0):.4f} | AUC {s.auc.mean():.4f} | {time.time()-t0:.0f}s", flush=True)
    scores = pd.concat(sc, ignore_index=True)
    oof = pd.DataFrame(oo); oof.insert(0, "SMILES", smi); oof.insert(1, "label", yd)
    scores.to_csv(RES / "cv_scores.csv", index=False); oof.to_csv(RES / "oof_predictions.csv", index=False)
else:
    scores = pd.read_csv(RES / "cv_scores.csv"); oof = pd.read_csv(RES / "oof_predictions.csv")
""")
md("## 3. 단독 비교")
code("""
s2 = pd.read_csv(M / "exp02_feature_variants/results/cv_scores.csv"); s3 = pd.read_csv(M / "exp03_tuning/results/cv_scores.csv")
pairs = {"seedavg|XGBoost": (s3, "tuned|XGBoost"), "seedavg|LightGBM": (s3, "tuned|LightGBM"),
         "transfer|+ap_tt|LightGBM": (s2, "+ap_tt|LightGBM"), "transfer|+ptfp|LightGBM": (s2, "+ptfp|LightGBM"),
         "transfer|+mordred|XGBoost": (s2, "+mordred|XGBoost"), "transfer|count_chiral|RandomForest": (s2, "count_chiral|RandomForest")}
rows = []
for new, (src, old) in pairs.items():
    a = src[src.model == old].sort_values(["repeat", "fold"]); b = scores[scores.model == new].sort_values(["repeat", "fold"])
    d = b.f1.values - a.f1.values
    rows.append((new, old, a.f1.mean(), b.f1.mean(), d.mean(), S.paired_p(d), int((d > 0).sum()), a.f1.std(ddof=0), b.f1.std(ddof=0)))
single = pd.DataFrame(rows, columns=["7차 모델", "비교 대상", "이전 F1", "7차 F1", "차이", "p값", "10 fold 중 개선", "이전 F1 std", "7차 F1 std"]).set_index("7차 모델")
display(single.round(4))
""")
md("## 4. 앙상블: 지금까지 최고 vs + 7차 모델")
code("""
pool = S.base_pool(smi); base33 = list(pool)
f6 = json.load(open(M / "exp06_knn_features/results/final_method.json"))
pool.update(S.read_oof(M / "exp06_knn_features/results/oof_predictions.csv", "6차:", smi))
best_prior_names = f6["models"] if f6["cv_f1"] > 0.8282 else base33
best_prior_label = "지금까지 최고: 6차 최종" if f6["cv_f1"] > 0.8282 else "지금까지 최고: 4차 최종"
C_prior = f6["C"] if f6["cv_f1"] > 0.8282 else 0.1
pool.update(S.read_oof(RES / "oof_predictions.csv", "7차:", smi))
new = [f"7차:{n}" for n in models7.NAMES]
methods = {"4차 최종: 스태킹 33개": S.m_stack(pool, base33, yd, C=0.1),
           best_prior_label: S.m_stack(pool, best_prior_names, yd, C=C_prior),
           "지금까지 최고 + 7차 6개": S.m_stack(pool, best_prior_names + new, yd, C=0.1),
           "지금까지 최고 + 7차 옮긴 설정 4개": S.m_stack(pool, best_prior_names + new[2:], yd, C=0.1),
           "평균: 7차 시드 평균 2개": S.m_mean(pool, new[:2])}
res = {n: S.evaluate(fn, yd, folds) for n, fn in methods.items()}
ref4 = res["4차 최종: 스태킹 33개"]["f1_thr"].values; refb = res[best_prior_label]["f1_thr"].values
rows = []
for n, r in res.items():
    d4 = r.f1_thr.values - ref4; db = r.f1_thr.values - refb
    rows.append((n, len(r), r.f1.mean(), r.f1_thr.mean(), r.f1_thr.std(ddof=0), r.auc.mean(), d4.mean(), S.paired_p(d4), db.mean(), S.paired_p(db), int((db > 0).sum())))
tbl = pd.DataFrame(rows, columns=["방법", "fold", "F1@0.5", "F1(fold 안 임계값)", "F1_std", "AUC", "4차 대비", "p(4차)", "지금까지 최고 대비", "p(최고)", "10 fold 중 개선(최고 대비)"]).drop(columns="fold")
tbl.to_csv(RES / "ensemble_compare.csv", index=False)
display(tbl.round(4))
""")
md("## 5. 7차 최종 방법 확정")
code("""
cand = tbl[tbl["방법"].str.contains("7차")].sort_values("F1(fold 안 임계값)", ascending=False).iloc[0]
METHOD = cand["방법"]
if METHOD.startswith("평균"):
    names = new[:2]
    p_full = np.mean([np.mean([pool[n]["r0"], pool[n]["r1"]], 0) for n in names], 0)
    final = {"method": METHOD, "kind": "mean", "models": names, "weights": [0.5, 0.5], "C": None, "intercept": None,
             "threshold": float(S.best_thr(yd, p_full)), "cv_f1": float(cand["F1(fold 안 임계값)"])}
    json.dump(final, open(RES / "final_method.json", "w"), ensure_ascii=False, indent=1)
else:
    names = best_prior_names + (new if "6개" in METHOD else new[2:])
    final = S.finalize(pool, names, yd, 0.1, RES / "final_method.json", METHOD, cand["F1(fold 안 임계값)"])
print(METHOD, "| 모델 수", len(names), "| 임계값", final["threshold"], "| CV F1", round(final["cv_f1"], 4))

long = pd.concat([r.assign(method=n) for n, r in res.items()])
en = {"4차 최종: 스태킹 33개": "S4 stack33", best_prior_label: "best so far", "지금까지 최고 + 7차 6개": "best + S7 all6",
      "지금까지 최고 + 7차 옮긴 설정 4개": "best + S7 transfer4", "평균: 7차 시드 평균 2개": "mean S7 seedavg2"}
long["method"] = long.method.map(en)
fig, ax = plt.subplots(figsize=(11, 4.3))
sns.boxplot(data=long, x="method", y="f1_thr", ax=ax, color="#cfe0f3"); sns.stripplot(data=long, x="method", y="f1_thr", ax=ax, color="#2b5d9a", size=4)
ax.set_xlabel(""); ax.set_ylabel("F1 (threshold chosen inside folds)"); ax.set_title("Seed averaging and transferred tuning (10 folds)")
plt.tight_layout(); plt.savefig(RES / "fig_seed_avg.png", dpi=130); plt.show()
""")
md("## 6. 결론")
r_ = open("README.md", encoding="utf-8").read()
md(r_[r_.index("## 판단"):r_.index("## 파일")].replace("\n## ", "\n### ").replace("## 판단", "### 판단"))
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "exp07_seed_avg.ipynb"); print("cells", len(C))
