import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 2차 실험: 입력 표현 바꾸기 (특성 블록 추가 · 교체 · 차원 축소)

| 항목 | 내용 |
|---|---|
| 목적 | 1차 상위 모델(RandomForest, LightGBM, XGBoost)을 고정하고, 입력 표현을 바꿔 F1이 오르는지 확인한다 |
| 기준(base) | Day1 1,042컬럼 — 1차 결과를 그대로 불러와 비교 (같은 fold, 같은 하이퍼파라미터) |
| 분할 | `../common/splits.csv` 고정 (개발 6,400행, 5-fold × 2회), 테스트 1,600행은 사용 안 함 |
| 비교 | 모델별로 base와 fold별 짝지은 보정 t-검정 |

| 변형 | 내용 | 근거 (1차 인사이트) |
|---|---|---|
| +rdkit217 | base + RDKit 전체 descriptor 217개 (정리 후) | 연속형 특성의 컬럼당 기여가 Fingerprint의 10~45배 |
| +mordred | base + Mordred 2D descriptor 1,613개 (정리 후) | 같은 근거, 더 넓은 물리·화학 정보 |
| +ptfp | base + PTFP 1,024비트 (정리 후) | Day1에서 고정 설정으로만 확인 |
| +maccs | base + MACCS 167비트 | 화학자가 정의한 부분구조 키, 다른 종류의 구조 정보 |
| +ap_tt | base + Atom-pair · Topological torsion count | 원자 간 거리·비틀림 정보, ECFP와 다른 관점 |
| count_chiral | ECFP·FCFP 비트를 count + 입체 정보 Fingerprint로 교체 | 처음 보는 구조에서 정답률 0.74, Day1 라벨 충돌은 입체 정보 때문 |
| combined | 위에서 효과가 있었던 블록을 모두 합침 | 자동 선택 (아래 규칙) |
| svd | Fingerprint 비트를 TruncatedSVD 256차원으로 줄임 (fold 안에서 fit) | 차원 축소 — 트리보다 SVM·MLP에서 효과 기대 |

블록 정리 규칙(라벨 미사용, 개발 세트로만 계산): 연속형은 결측 20% 초과·상수·기존 연속형과 |상관| ≥ 0.95 제거, Fingerprint형은 0이 아닌 비율 2~98%만 남기고 완전 중복 제거.

실행 방법: `build_features.py`를 먼저 한 번 실행해 `cache/`를 만든다(약 10분). 이후 노트북을 위에서부터 실행(약 25~30분). `RUN_CV = False`면 저장된 결과만 불러온다.
""")

code("""
import sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb, xgboost as xgb

HERE = Path.cwd()
sys.path.insert(0, str(HERE.parent / "common"))
from preprocess import preprocess_train, load_feature_columns
from evaluate import load_splits, run_cv, summarize, corrected_paired_ttest
import featurize as F

warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
RES = HERE / "results"; RES.mkdir(exist_ok=True)
CACHE = HERE / "cache"
EXP01 = HERE.parent / "exp01_model_comparison" / "results"
RUN_CV = True
SEED, N_JOBS = 0, 2
""")

md("## 1. 데이터 · 분할 · 특성 블록 불러오기")
code("""
df, X, y, groups = preprocess_train(HERE.parent.parent / "train.csv")
cols = load_feature_columns()
df = load_splits(df)
idx = pd.read_parquet(CACHE / "index.parquet")
assert (idx["SMILES"].values == df["SMILES"].values).all(), "cache 순서가 다름 — build_features.py 재실행"

dev = (df["set"] == "dev").values
yd = y[dev]
folds = df.loc[dev, ["fold_r0", "fold_r1"]].reset_index(drop=True)
base = X[dev].reset_index(drop=True).astype("float32")
fp_base = [c for c in cols if c.startswith(("ecfp_", "fcfp_"))]
cont_base = [c for c in cols if c not in fp_base and c not in ("is_salt", "has_stereo_tag")]

raw = {b: F.read_cache(CACHE, b)[dev].reset_index(drop=True)
       for b in ["rdkit217", "mordred", "ptfp", "maccs", "ap_tt", "ecfp_cnt", "fcfp_cnt"]}
print("개발 세트:", base.shape, {k: v.shape[1] for k, v in raw.items()})
""")

code("""
blocks, spec, rows = {}, {}, []
for b, d in raw.items():
    kind = "cont" if b in F.CONT_BLOCKS else "fp"
    keep, med = F.filter_block(d, kind, ref_cont=base[cont_base] if kind == "cont" else None)
    blocks[b] = F.apply_filter(d, keep, med).astype("float32")
    spec[b] = {"kind": kind, "columns": keep, "median": med}
    rows.append((b, kind, d.shape[1], len(keep)))
json.dump(spec, open(RES / "block_spec.json", "w"))
display(pd.DataFrame(rows, columns=["블록", "종류", "원래 컬럼 수", "정리 후"]).set_index("블록"))
""")

md("## 2. 변형 정의와 모델")
code("""
nonfp_base = [c for c in cols if c not in fp_base]
variants = {
    "+rdkit217": pd.concat([base, blocks["rdkit217"]], axis=1),
    "+mordred": pd.concat([base, blocks["mordred"]], axis=1),
    "+ptfp": pd.concat([base, blocks["ptfp"]], axis=1),
    "+maccs": pd.concat([base, blocks["maccs"]], axis=1),
    "+ap_tt": pd.concat([base, blocks["ap_tt"]], axis=1),
    "count_chiral": pd.concat([blocks["ecfp_cnt"], blocks["fcfp_cnt"], base[nonfp_base]], axis=1),
}
print({k: v.shape[1] for k, v in variants.items()})

ALGOS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=N_JOBS, random_state=SEED),
    "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3,
                                           subsample=0.8, subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=N_JOBS),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                         colsample_bytree=0.3, tree_method="hist", eval_metric="logloss",
                                         random_state=SEED, n_jobs=N_JOBS),
}
""")

md("## 3. 교차검증 (변형 × 모델)")
code("""
def run_variant(vname, Xv, algos, store):
    for a, mk in algos.items():
        t0 = time.time()
        s, o = run_cv(mk, Xv, yd, folds, f"{vname}|{a}")
        store["scores"].append(s)
        for r in o.columns:
            store["oof"][f"{vname}|{a}|{r}"] = o[r].values
        print(f"{vname:14s} {a:12s} F1 {s.f1.mean():.4f} ± {s.f1.std(ddof=0):.4f} | AUC {s.auc.mean():.4f} | {time.time()-t0:4.0f}s", flush=True)

# base는 1차 결과를 재사용 (같은 fold·같은 하이퍼파라미터)
s1 = pd.read_csv(EXP01 / "cv_scores.csv"); o1 = pd.read_csv(EXP01 / "oof_predictions.csv")
base_scores = s1[s1.model.isin(list(ALGOS) + ["SVM_RBF", "MLP"])].copy()
base_scores["model"] = "base|" + base_scores["model"]
base_oof = {f"base|{c}": o1[c].values for c in o1.columns if "|" in c and c.split("|")[0] in list(ALGOS) + ["SVM_RBF", "MLP"]}

if RUN_CV:
    store = {"scores": [base_scores], "oof": dict(base_oof)}
    for vname, Xv in variants.items():
        run_variant(vname, Xv, ALGOS, store)
""")

md("""
### 3-1. combined: 효과가 있었던 블록 합치기
- 규칙: 세 모델의 base 대비 F1 변화 평균이 +0.002 이상인 추가 블록을 모두 합친다.
- count_chiral이 기준을 넘으면 ECFP·FCFP 비트를 count 버전으로 바꾼 상태에서 합친다.
""")
code("""
if RUN_CV:
    sc = pd.concat(store["scores"], ignore_index=True)
    sc[["variant", "algo"]] = sc.model.str.split("|", expand=True)
    mean_f1 = sc.groupby(["variant", "algo"]).f1.mean().unstack()
    delta = (mean_f1 - mean_f1.loc["base"]).mean(axis=1).drop("base")
    good = [v for v in delta.index if delta[v] >= 0.002]
    print("base 대비 평균 ΔF1:", delta.round(4).to_dict()); print("합칠 변형:", good)
    use_cnt = "count_chiral" in good
    parts = [blocks["ecfp_cnt"], blocks["fcfp_cnt"], base[nonfp_base]] if use_cnt else [base]
    parts += [blocks[v[1:]] for v in good if v.startswith("+")]
    if len(good) >= 2:
        variants["combined"] = pd.concat(parts, axis=1)
        print("combined 컬럼 수:", variants["combined"].shape[1])
        run_variant("combined", variants["combined"], ALGOS, store)
    json.dump({"good": good, "use_count_chiral": use_cnt}, open(RES / "combined_rule.json", "w"))
""")

md("""
### 3-2. 차원 축소 (TruncatedSVD)
- Fingerprint 비트 1,027개를 256차원으로 줄이고, 연속형 15개는 표준화해 그대로 붙인다. SVD와 표준화 모두 fold 안에서 fit한다.
- 트리 모델은 원래 비트 단위 분할에 강해 효과가 적을 것으로 보고, SVM·MLP(1차 6·7위)와 LightGBM을 비교한다.
""")
code("""
def svd_pipe(est):
    pre = ColumnTransformer([("svd", TruncatedSVD(256, random_state=SEED), fp_base),
                             ("sc", StandardScaler(), [c for c in cols if c not in fp_base])])
    return make_pipeline(pre, est)

SVD_ALGOS = {
    "SVM_RBF": lambda: svd_pipe(SVC(C=1.0, kernel="rbf", gamma="scale", random_state=SEED)),
    "MLP": lambda: svd_pipe(MLPClassifier((128,), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED)),
    "LightGBM": lambda: svd_pipe(lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3,
                                                    subsample=0.8, subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=N_JOBS)),
}
if RUN_CV:
    run_variant("svd", base, SVD_ALGOS, store)
    scores = pd.concat(store["scores"], ignore_index=True)
    oof = pd.DataFrame(store["oof"]); oof.insert(0, "SMILES", df.loc[dev, "SMILES"].values); oof.insert(1, "label", yd)
    scores.to_csv(RES / "cv_scores.csv", index=False); oof.to_csv(RES / "oof_predictions.csv", index=False)
else:
    scores = pd.read_csv(RES / "cv_scores.csv"); oof = pd.read_csv(RES / "oof_predictions.csv")
scores[["variant", "algo"]] = scores.model.str.split("|", expand=True)
""")

md("## 4. 결과")
code("""
f1_tbl = scores.groupby(["variant", "algo"]).f1.mean().unstack()
auc_tbl = scores.groupby(["variant", "algo"]).auc.mean().unstack()
ncol = {"base": len(cols), **{k: v.shape[1] for k, v in variants.items()}, "svd": 256 + 15}
order_v = ["base"] + [v for v in f1_tbl.index if v != "base"]
f1_tbl = f1_tbl.loc[order_v]; auc_tbl = auc_tbl.loc[order_v]

# 모델별 base 대비 짝지은 검정
tests = []
for a in f1_tbl.columns:
    sub = scores[scores.algo == a].copy(); sub["model"] = sub["variant"]
    if "base" not in set(sub.model): continue
    t = corrected_paired_ttest(sub, "base"); t["algo"] = a; tests.append(t.reset_index())
tests = pd.concat(tests).rename(columns={"model": "variant"})
tests.to_csv(RES / "paired_ttest_vs_base.csv", index=False)

show = f1_tbl.copy()
show.insert(0, "컬럼 수", [ncol.get(v, np.nan) for v in show.index])
show["트리 3개 평균"] = f1_tbl[["RandomForest", "LightGBM", "XGBoost"]].mean(axis=1)
show.to_csv(RES / "f1_by_variant.csv")
print("F1 (10 fold 평균)")
display(show.round(4).style.background_gradient(subset=["트리 3개 평균"], cmap="Blues").format(precision=4, na_rep="-"))
print("base 대비 차이와 보정 짝지은 t-검정 p값")
piv = tests.pivot(index="variant", columns="algo", values="diff_vs_best").round(4).astype(str) + " (p=" + \\
      tests.pivot(index="variant", columns="algo", values="p_value").round(3).astype(str) + ")"
display(piv.loc[[v for v in order_v if v in piv.index]])
""")

md("## 5. 시각화")
code("""
d = (f1_tbl - f1_tbl.loc["base"]).drop("base")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
sns.heatmap(d, annot=True, fmt="+.4f", cmap="RdBu", center=0, vmin=-0.012, vmax=0.012, ax=ax[0], cbar_kws={"label": "ΔF1 vs base"})
ax[0].set_title("F1 change vs base (same model, same folds)"); ax[0].set_xlabel(""); ax[0].set_ylabel("")
m = scores[scores.algo.isin(["RandomForest", "LightGBM", "XGBoost"])]
order_bar = f1_tbl[["RandomForest", "LightGBM", "XGBoost"]].mean(axis=1).dropna().sort_values(ascending=False).index
sns.barplot(data=m, x="variant", y="f1", hue="algo", order=order_bar, ax=ax[1], errorbar="sd")
ax[1].set_ylim(0.79, 0.84); ax[1].set_xlabel(""); ax[1].set_title("F1 by variant (mean ± sd, 10 folds)"); ax[1].tick_params(axis="x", rotation=30)
ax[1].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(RES / "fig_variant_f1.png", dpi=130); plt.show()
""")

md("""
### 5-1. 처음 보는 구조에서 나아졌는가
1차에서 학습 데이터와 유사도가 낮을수록 정답률이 떨어졌다. 가장 좋은 변형이 그 구간을 개선했는지 본다.
""")
code("""
tree_mean = f1_tbl[["RandomForest", "LightGBM", "XGBoost"]].mean(axis=1).drop("base")
best_v = tree_mean.idxmax()
best_a = f1_tbl.loc[best_v, ["RandomForest", "LightGBM", "XGBoost"]].idxmax()
print("가장 좋은 변형:", best_v, "/", best_a)
# 3·4차에서 쓰도록 변형별 컬럼 목록과 최고 변형 저장 (svd는 파이프라인이라 base 컬럼)
vc = {"base": list(cols), **{k: list(v.columns) for k, v in variants.items()}}
best_nonsvd = tree_mean.drop("svd", errors="ignore").idxmax()
json.dump({"best_variant": best_nonsvd, "tree_mean_f1": tree_mean.round(4).to_dict(), "columns": vc},
          open(RES / "variant_columns.json", "w"))

E = df.loc[dev, [f"ecfp_{i}" for i in range(1024)]].values.astype(np.float32); cnt = E.sum(1)
max_sim = np.zeros(len(E)); f0 = folds["fold_r0"].values
for k in range(5):
    va, tr = f0 == k, f0 != k
    inter = E[va] @ E[tr].T
    max_sim[va] = (inter / (cnt[va][:, None] + cnt[tr][None, :] - inter + 1e-9)).max(1)
lab = ["<0.5", "0.5-0.7", "0.7-0.8", "≥0.8"]
b = pd.cut(max_sim, [0, 0.5, 0.7, 0.8, 1.01], labels=lab, right=False)
acc = pd.DataFrame({
    f"base|{best_a}": [((oof[f"base|{best_a}|r0"] >= .5).astype(int) == yd)[b == l].mean() for l in lab],
    f"{best_v}|{best_a}": [((oof[f"{best_v}|{best_a}|r0"] >= .5).astype(int) == yd)[b == l].mean() for l in lab],
    "화합물 수": [int((b == l).sum()) for l in lab]}, index=lab)
acc["차이"] = acc.iloc[:, 1] - acc.iloc[:, 0]
display(acc.round(3))
""")

md("""
### 5-2. 앙상블 재료로서의 가치 (4차 미리보기)
- 다른 입력으로 학습한 모델이 base 모델과 얼마나 다르게 예측하는지(순위 상관), 그리고 둘을 평균하면 F1이 어떻게 되는지 본다.
""")
code("""
ref = f"base|{best_a}"
rows = []
for c in sorted({c.rsplit("|", 1)[0] for c in oof.columns if "|" in c}):
    if c == ref: continue
    p0 = np.mean([oof[f"{c}|r{r}"] for r in (0, 1)], axis=0)
    rho = pd.Series(oof[f"{c}|r0"]).corr(pd.Series(oof[f"{ref}|r0"]), method="spearman")
    single = np.mean([f1_score(yd, oof[f"{c}|r{r}"] >= .5) for r in (0, 1)])
    pair = np.mean([f1_score(yd, (oof[f"{c}|r{r}"] + oof[f"{ref}|r{r}"]) / 2 >= .5) for r in (0, 1)]) if "SVM" not in c else np.nan
    rows.append((c, rho, single, pair))
ens = pd.DataFrame(rows, columns=["모델", f"{ref}와 순위 상관", "단독 F1", f"{ref}와 평균 F1"]).set_index("모델")
ref_f1 = np.mean([f1_score(yd, oof[f"{ref}|r{r}"] >= .5) for r in (0, 1)])
print(f"{ref} 단독 F1: {ref_f1:.4f}")
display(ens.sort_values(f"{ref}와 평균 F1", ascending=False).round(4))
""")

md("## 6. 결론과 3차로 넘길 것")
r_ = open("README.md", encoding="utf-8").read()
md(r_[r_.index("## 판단"):r_.index("## 파일")].replace("\n## ", "\n### ").replace("## 판단", "### 판단"))

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "exp02_feature_variants.ipynb")
print("cells:", len(C))
