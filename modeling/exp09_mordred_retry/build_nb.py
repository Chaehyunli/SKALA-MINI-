import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 9차 실험: Mordred 재도전

| 항목 | 내용 |
|---|---|
| 목표 | 지금까지 최고 스태킹을 넘는다 |
| 2차 결과 | base + Mordred는 단독으로 효과가 없었지만(트리 평균 −0.002), +Mordred XGBoost는 4차 스태킹 가중치 5위였다 — 앙상블 재료로는 쓸모가 있었다 |
| ① descriptor만 쓰는 모델 | Fingerprint 없이 Descriptor 3 + RDKit 12 + RDKit 217(168) + Mordred(700) = 883개만 사용. 1~3차와 전혀 다른 관점이라 앙상블에서 다르게 틀릴 것으로 기대 (LightGBM, XGBoost) |
| ② Mordred + 튜닝 | base + Mordred(1,742컬럼)에 3차 튜닝 LightGBM 설정 적용 (2차는 기본 설정) |
| 비교 | 단독 / 앙상블: 지금까지 최고 스태킹 + 9차 모델 (같은 10 fold, fold 안 임계값) |

실행 약 20분.
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
import models9, stacking as S

warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
RES = HERE / "results"; RES.mkdir(exist_ok=True)
RUN_CV = True; STAGE = "9차"
""")
code("""
df, X, y, groups = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
ALL = load_all(df, X); dev = (df.set == "dev").values
Xd = ALL[dev].reset_index(drop=True); yd = y[dev]; smi = df.loc[dev, "SMILES"].values
folds = df.loc[dev, ["fold_r0", "fold_r1"]].reset_index(drop=True)
print("개발 세트:", Xd.shape)
""")
md("## 1. 9차 모델 10-fold 교차검증")
code("""
if RUN_CV:
    sc, oo = [], {}
    for name in models9.NAMES:
        mk, use = models9.build(name); t0 = time.time()
        s, o = run_cv(mk, Xd[use], yd, folds, name)
        sc.append(s)
        for r in o.columns: oo[f"{name}|{r}"] = o[r].values
        print(f"{name:22s} 컬럼 {len(use):5d} | F1 {s.f1.mean():.4f} ± {s.f1.std(ddof=0):.4f} | AUC {s.auc.mean():.4f} | {time.time()-t0:.0f}s", flush=True)
    scores = pd.concat(sc, ignore_index=True)
    oof = pd.DataFrame(oo); oof.insert(0, "SMILES", smi); oof.insert(1, "label", yd)
    scores.to_csv(RES / "cv_scores.csv", index=False); oof.to_csv(RES / "oof_predictions.csv", index=False)
else:
    scores = pd.read_csv(RES / "cv_scores.csv"); oof = pd.read_csv(RES / "oof_predictions.csv")
""")
md("## 2. 단독 성능과 다른 모델과의 예측 상관")
code("""
display(scores.groupby("model")[["f1", "auc", "precision", "recall"]].mean().round(4))
s2 = pd.read_csv(M / "exp02_feature_variants/results/cv_scores.csv")
a = s2[s2.model == "+mordred|LightGBM"].sort_values(["repeat", "fold"]); b = scores[scores.model == "mordredtuned|LightGBM"].sort_values(["repeat", "fold"])
d = b.f1.values - a.f1.values
print(f"Mordred + LightGBM: 기본 설정 {a.f1.mean():.4f} → 튜닝 설정 {b.f1.mean():.4f} ({d.mean():+.4f}, p={S.paired_p(d):.3f})")
""")
md("## 3. 앙상블: 지금까지 최고 vs + 9차 모델")
code("""
pool = S.full_pool(smi, STAGE); base33 = list(S.base_pool(smi))
label, prior, Cp, prior_f1 = S.best_so_far(STAGE)
pool.update(S.read_oof(RES / "oof_predictions.csv", f"{STAGE}:", smi))
new = [f"{STAGE}:{n}" for n in models9.NAMES]
methods = {"4차 최종: 스태킹 33개": S.m_stack(pool, base33, yd, C=0.1), label: S.m_stack(pool, prior, yd, C=Cp),
           "지금까지 최고 + 9차 3개": S.m_stack(pool, prior + new, yd, C=0.1)}
res = {n: S.evaluate(fn, yd, folds) for n, fn in methods.items()}
ref4 = res["4차 최종: 스태킹 33개"]["f1_thr"].values; refb = res[label]["f1_thr"].values
rows = []
for n, r in res.items():
    d4 = r.f1_thr.values - ref4; db = r.f1_thr.values - refb
    rows.append((n, r.f1.mean(), r.f1_thr.mean(), r.f1_thr.std(ddof=0), r.auc.mean(), d4.mean(), S.paired_p(d4), db.mean(), S.paired_p(db), int((db > 0).sum())))
tbl = pd.DataFrame(rows, columns=["방법", "F1@0.5", "F1(fold 안 임계값)", "F1_std", "AUC", "4차 대비", "p(4차)", "지금까지 최고 대비", "p(최고)", "10 fold 중 개선(최고 대비)"])
tbl.to_csv(RES / "ensemble_compare.csv", index=False); display(tbl.round(4))
""")
md("## 4. 9차 최종 방법 확정")
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
nbf.write(nb, "exp09_mordred_retry.ipynb"); print("cells", len(C))
