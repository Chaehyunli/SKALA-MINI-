import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 5차 실험: 그래프 신경망 (Chemprop D-MPNN)

| 항목 | 내용 |
|---|---|
| 목적 | 1~4차 모델은 모두 Fingerprint·descriptor 기반이라 비슷하게 틀린다. 분자 그래프를 직접 학습하는 모델을 만들어 단독 성능과 앙상블 기여를 확인한다 |
| 모델 | Chemprop 2 D-MPNN — 원자를 노드, 결합을 엣지로 보고 결합 방향으로 메시지를 3단계 전달(은닉 300) → 분자 벡터 평균 → 분자 단위 특성 15개(Descriptor 3 + RDKit 12, 표준화)를 붙여 분류 |
| 학습 | 25 epoch 고정, 배치 64, dropout 0.1. 검증 fold는 학습 중에 보지 않는다 |
| 분할 | 1~4차와 같은 `common/splits.csv` 10 fold — OOF 예측을 4차 스태킹 후보에 바로 추가할 수 있다 |
| 비교 | ① GNN 단독 vs 3차 최고 단일 모델 ② 4차 최종 스태킹 vs 같은 스태킹 + GNN (fold별 짝지은 보정 t-검정) |

epoch 수는 시간 제약(CPU 2코어, epoch당 약 15초)으로 25로 고정했다. 5 epoch만으로도 fold 0에서 F1 0.781·AUC 0.823이었다.
실행 시간: GNN 교차검증 약 60분. `RUN_GNN = False`면 저장된 OOF를 불러온다.
""")
code("""
import sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

HERE = Path.cwd(); M = HERE.parent
sys.path.insert(0, str(M / "common"))
from preprocess import preprocess_train
from evaluate import load_splits
import gnn, ensemble as E, stacking as S

warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
RES = HERE / "results"; RES.mkdir(exist_ok=True)
RUN_GNN = True
EPOCHS = 25
XD_COLS = ["MolWt", "clogp", "sa_score", "TPSA", "HBD", "HBA", "n_rotb", "n_ring", "n_arom_ring", "frac_csp3",
           "n_halogen", "n_S", "abs_charge", "is_salt", "has_stereo_tag"]
""")
md("## 1. 데이터")
code("""
df, X, y, groups = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
d = df[df.set == "dev"].reset_index(drop=True)
yd = d["label"].values; smi = d["SMILES"].values
folds = d[["fold_r0", "fold_r1"]]
print("개발 세트:", len(d), "| 분자 단위 특성:", len(XD_COLS))
""")
md("## 2. GNN 10-fold 교차검증")
code("""
if RUN_GNN:
    rows, oof = [], {}
    for ri, r in enumerate(("r0", "r1")):
        f = folds[f"fold_r{ri}"].values; p_oof = np.zeros(len(d))
        for k in range(5):
            tr, va = f != k, f == k; t0 = time.time()
            p = gnn.fit_predict(smi[tr].tolist(), yd[tr], smi[va].tolist(), d.loc[tr, XD_COLS].values, d.loc[va, XD_COLS].values,
                                epochs=EPOCHS, seed=ri)
            p_oof[va] = p
            rows.append(dict(model="gnn", repeat=ri, fold=k, f1=f1_score(yd[va], p >= .5), auc=roc_auc_score(yd[va], p),
                             precision=precision_score(yd[va], p >= .5), recall=recall_score(yd[va], p >= .5),
                             fit_sec=time.time() - t0, n_train=int(tr.sum()), n_val=int(va.sum())))
            print(f"repeat {ri} fold {k}: F1 {rows[-1]['f1']:.4f} AUC {rows[-1]['auc']:.4f} ({rows[-1]['fit_sec']:.0f}s)", flush=True)
        oof[f"gnn|{r}"] = p_oof
    scores = pd.DataFrame(rows)
    oof = pd.DataFrame(oof); oof.insert(0, "SMILES", smi); oof.insert(1, "label", yd)
    scores.to_csv(RES / "cv_scores.csv", index=False); oof.to_csv(RES / "oof_predictions.csv", index=False)
else:
    scores = pd.read_csv(RES / "cv_scores.csv"); oof = pd.read_csv(RES / "oof_predictions.csv")
print(f"GNN F1 {scores.f1.mean():.4f} ± {scores.f1.std(ddof=0):.4f} | AUC {scores.auc.mean():.4f}")
""")
md("## 3. 1~9차 모델과 함께 보기")
code("""
pool = S.full_pool(smi, "없음")          # 1~4차 + 6~9차 OOF 전부
base33 = list(S.base_pool(smi))
label, prior, Cp, prior_f1 = S.best_so_far("없음")
pool.update(E.read_oof(RES / "oof_predictions.csv", "5차:", smi))
GNN = "5차:gnn"; BEST = "3차:tuned|XGBoost"
print("지금까지 최고:", label, "| CV F1", round(prior_f1, 4), "| 모델 수", len(prior))
single = pd.DataFrame({k: [np.mean([f1_score(yd, pool[k][r] >= .5) for r in ("r0", "r1")]), np.mean([roc_auc_score(yd, pool[k][r]) for r in ("r0", "r1")]),
                           pd.Series(pool[k]["r0"]).corr(pd.Series(pool[BEST]["r0"]), method="spearman")]
                       for k in [GNN, BEST, "3차:tuned|LightGBM", "7차:transfer|count_chiral|RandomForest", "8차:pseudo|XGBoost", "1차:MLP", "1차:SVM_RBF"]},
                      index=["F1@0.5", "AUC", f"{BEST}와 순위 상관"]).T
display(single.round(4))
""")
code("""
# 단독 비교: GNN vs 3차 최고 단일
sc3 = pd.read_csv(M / "exp03_tuning/results/cv_scores.csv")
b = sc3[sc3.model == "tuned|XGBoost"].sort_values(["repeat", "fold"]).f1.values
g = scores.sort_values(["repeat", "fold"]).f1.values
print(f"GNN − 3차 XGBoost: {np.mean(g - b):+.4f} (p={E.paired_p(g - b):.3f}, 10 fold 중 GNN이 높은 횟수 {int((g > b).sum())})")

P = pd.DataFrame({k.replace("1차:", "S1 ").replace("2차:", "S2 ").replace("3차:", "S3 ").replace("5차:", "S5 "): pool[k]["r0"]
                  for k in [GNN, BEST, "3차:tuned|LightGBM", "3차:tuned|RandomForest", "2차:count_chiral|RandomForest", "2차:+mordred|XGBoost", "1차:MLP", "1차:SVM_RBF"]})
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(P.corr(method="spearman"), annot=True, fmt=".2f", cmap="viridis", vmin=0.6, vmax=1, ax=ax, annot_kws={"size": 8})
ax.set_title("Spearman correlation of OOF predictions (S5 gnn = graph neural net)"); plt.tight_layout(); plt.savefig(RES / "fig_pred_corr.png", dpi=130); plt.show()
""")
md("""
## 4. 앙상블에 GNN 추가 (같은 fold 안 평가)
- 기준: 지금까지 최고 스태킹(8차 최종, 41개)과 4차 최종(33개)
""")
code("""
methods = {"4차 최종: 스태킹 33개": E.m_stack(pool, base33, yd, C=0.1),
           label: E.m_stack(pool, prior, yd, C=Cp),
           "지금까지 최고 + GNN": E.m_stack(pool, prior + [GNN], yd, C=0.1),
           "4차 33개 + GNN": E.m_stack(pool, base33 + [GNN], yd, C=0.1),
           "GNN 단독": E.m_single(pool, GNN)}
res = {n: E.evaluate(fn, yd, folds) for n, fn in methods.items()}
ref4 = res["4차 최종: 스태킹 33개"]["f1_thr"].values; refb = res[label]["f1_thr"].values
rows = []
for n, r in res.items():
    d4 = r.f1_thr.values - ref4; db = r.f1_thr.values - refb
    rows.append((n, r.f1.mean(), r.f1_thr.mean(), r.f1_thr.std(ddof=0), r.auc.mean(), d4.mean(), E.paired_p(d4), db.mean(), E.paired_p(db), int((db > 0).sum()), r.thr.median()))
summary = pd.DataFrame(rows, columns=["방법", "F1@0.5", "F1(fold 안 임계값)", "F1_std", "AUC", "4차 대비", "p(4차)", "지금까지 최고 대비", "p(최고)", "10 fold 중 개선(최고 대비)", "임계값(중앙값)"])
summary.to_csv(RES / "summary.csv", index=False)
display(summary.round(4))
""")
code("""
long = pd.concat([r.assign(method=n) for n, r in res.items()])
name_en = {"4차 최종: 스태킹 33개": "S4 stack33", label: "best so far (S8)", "지금까지 최고 + GNN": "best + GNN", "4차 33개 + GNN": "S4 stack33 + GNN", "GNN 단독": "GNN alone"}
long["method"] = long.method.map(name_en)
fig, ax = plt.subplots(figsize=(11, 4.3))
sns.boxplot(data=long, x="method", y="f1_thr", ax=ax, color="#cfe0f3"); sns.stripplot(data=long, x="method", y="f1_thr", ax=ax, color="#2b5d9a", size=4)
ax.set_xlabel(""); ax.set_ylabel("F1 (threshold chosen inside folds)"); ax.set_title("Adding the graph neural net to the ensemble (10 folds)")
plt.tight_layout(); plt.savefig(RES / "fig_ensemble_gnn.png", dpi=130); plt.show()
""")
md("""
## 5. 5차 최종 방법 확정 (제출 파일에 사용)
- GNN이 들어간 방법 중 F1(fold 안 임계값)이 가장 높은 것을 5차 최종으로 한다.
- 구성(스태킹 계수·임계값)은 개발 세트 전체 OOF(반복 0·1)로 다시 정해 `results/final_method.json`에 저장한다.
""")
code("""
cand = summary[summary["방법"].isin(["지금까지 최고 + GNN", "4차 33개 + GNN"])].sort_values("F1(fold 안 임계값)", ascending=False).iloc[0]
METHOD = cand["방법"]; names = prior + [GNN] if METHOD == "지금까지 최고 + GNN" else base33 + [GNN]
final = S.finalize(pool, names, yd, 0.1, RES / "final_method.json", METHOD, cand["F1(fold 안 임계값)"],
                   extra={"gnn": {"epochs": EPOCHS, "xd_cols": XD_COLS}})
w = pd.Series(final["weights"], index=names).sort_values(ascending=False)
print(METHOD, "| 모델 수", len(names), "| 임계값", final["threshold"], "| CV F1", round(final["cv_f1"], 4))
print("GNN 스태킹 가중치:", round(w[GNN], 3), "| 순위", list(w.index).index(GNN) + 1, "/", len(w))
""")
md("## 6. 결론")
md("(실행 결과를 보고 작성 — README.md와 같은 내용)")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "exp05_gnn.ipynb"); print("cells", len(C))
