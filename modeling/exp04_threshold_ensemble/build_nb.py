import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 4차 실험: 임계값 조정과 앙상블

| 항목 | 내용 |
|---|---|
| 목적 | 1~3차에서 만든 모델들의 OOF 예측을 조합하고 임계값을 조정해 F1을 더 올린다 (재학습 없음) |
| 재료 | 1차 9개 모델, 2차 변형별 모델, 3차 튜닝 모델의 OOF 예측 확률 — 모두 같은 개발 세트 6,400행·같은 fold |
| 평가 | 모든 선택(임계값, 가중치, 모델 고르기)을 fold 안에서 한다: fold k의 점수는 나머지 4개 fold의 OOF로만 정한 규칙을 fold k에 적용해서 잰다 |
| 비교 | 3차 최고 단일 모델(임계값 0.5)과 fold별 짝지은 보정 t-검정 |

| 방법 | 내용 |
|---|---|
| 임계값 조정 | 나머지 fold에서 F1이 최대가 되는 임계값을 골라 적용 |
| 단순 평균 | 3차 튜닝 모델 3개의 확률 평균 |
| 스태킹 | 각 모델 예측(logit)을 입력으로 하는 로지스틱 회귀가 가중치를 학습 |
| 탐욕적 선택 (Caruana) | 후보 중 F1을 가장 올리는 모델을 하나씩 더해 가며(중복 허용) 평균 — 나머지 fold에서 고름 |

참고: 스태킹·탐욕적 선택에서 fold j의 OOF를 만든 모델은 fold k 데이터로도 학습했다. 이 때문에 아주 약간 낙관적일 수 있으며, 일반적으로 허용되는 수준이다. 최종 확인은 테스트 세트로 한다.
""")

code("""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats

HERE = Path.cwd(); M = HERE.parent
sys.path.insert(0, str(M / "common"))
from preprocess import preprocess_train
from evaluate import load_splits

warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
RES = HERE / "results"; RES.mkdir(exist_ok=True)
""")

md("## 1. OOF 예측 모으기")
code("""
df, X, y, groups = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
dev = (df["set"] == "dev").values
yd = y[dev]; smi = df.loc[dev, "SMILES"].values
folds = df.loc[dev, ["fold_r0", "fold_r1"]].reset_index(drop=True)

def read_oof(path, prefix):
    o = pd.read_csv(path); assert (o["SMILES"].values == smi).all()
    out = {}
    for c in o.columns:
        if "|r" not in c: continue
        name, r = c.rsplit("|", 1)
        out.setdefault(prefix + name, {})[r] = o[c].values
    return out

pool = {}
pool.update(read_oof(M / "exp01_model_comparison/results/oof_predictions.csv", "1차:"))
pool.update({k: v for k, v in read_oof(M / "exp02_feature_variants/results/oof_predictions.csv", "2차:").items() if not k.startswith("2차:base|")})
pool.update(read_oof(M / "exp03_tuning/results/oof_predictions.csv", "3차:"))
pool = {k: v for k, v in pool.items() if set(v) == {"r0", "r1"}}
NONPROB = [k for k in pool if "SVM" in k]          # 확률이 아닌 점수 (평균에는 제외, 스태킹에는 사용)
print("후보 모델 수:", len(pool))

def f1_at(p, t=0.5): return f1_score(yd_, p >= t)
single = pd.DataFrame({k: [np.mean([f1_score(yd, v[r] >= .5) for r in ("r0", "r1")]),
                           np.mean([roc_auc_score(yd, v[r]) for r in ("r0", "r1")])] for k, v in pool.items()},
                      index=["F1@0.5", "AUC"]).T.sort_values("F1@0.5", ascending=False)
display(single.head(15).round(4))
tuned = [k for k in pool if k.startswith("3차:")]
BEST = max(tuned, key=lambda k: single.loc[k, "F1@0.5"])
print("기준(3차 최고 단일 모델):", BEST)
""")

md("## 2. fold 안에서 규칙을 정하는 평가 함수")
code("""
ths = np.round(np.arange(0.30, 0.701, 0.01), 2)
def best_thr(yv, p):
    f = [f1_score(yv, p >= t) for t in ths]; return ths[int(np.argmax(f))]
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def evaluate(method):
    \"\"\"method(r, tr_mask, va_mask) -> (검증 fold 확률, 학습 fold 확률). 반환: fold별 F1(0.5), F1(fold 안 임계값), 임계값\"\"\"
    rows = []
    for ri, r in enumerate(("r0", "r1")):
        f = folds[f"fold_r{ri}"].values
        for k in range(5):
            tr, va = f != k, f == k
            p_va, p_tr = method(r, tr, va)
            t = best_thr(yd[tr], p_tr)
            rows.append(dict(repeat=ri, fold=k, f1=f1_score(yd[va], p_va >= .5), f1_thr=f1_score(yd[va], p_va >= t),
                             thr=t, auc=roc_auc_score(yd[va], p_va)))
    return pd.DataFrame(rows)

def m_single(name):
    return lambda r, tr, va: (pool[name][r][va], pool[name][r][tr])
def m_mean(names):
    return lambda r, tr, va: (np.mean([pool[n][r][va] for n in names], 0), np.mean([pool[n][r][tr] for n in names], 0))
def m_stack(names, C=1.0):
    def f(r, tr, va):
        Z = np.column_stack([logit(pool[n][r]) for n in names])
        lr = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], yd[tr])
        return lr.predict_proba(Z[va])[:, 1], lr.predict_proba(Z[tr])[:, 1]
    return f
def m_greedy(cands, n_iter=15):
    def f(r, tr, va):
        chosen, cur = [], None
        for _ in range(n_iter):
            best, bf = None, -1
            for c in cands:
                p = pool[c][r][tr] if cur is None else (cur * len(chosen) + pool[c][r][tr]) / (len(chosen) + 1)
                s = f1_score(yd[tr], p >= .5)
                if s > bf: best, bf = c, s
            chosen.append(best)
            cur = np.mean([pool[c][r][tr] for c in chosen], 0)
        f.last = chosen
        return np.mean([pool[c][r][va] for c in chosen], 0), cur
    return f
""")

md("## 3. 방법별 평가")
code("""
prob_all = [k for k in pool if k not in NONPROB]
top_prob = single.loc[prob_all].sort_values("F1@0.5", ascending=False).index[:12].tolist()
methods = {
    f"단일: {BEST}": m_single(BEST),
    "평균: 3차 튜닝 3개": m_mean(tuned),
    "스태킹: 3차 튜닝 3개": m_stack(tuned),
    "스태킹: 전체 후보": m_stack(list(pool), C=0.1),
    "탐욕적 선택: 상위 12개 후보": m_greedy(top_prob),
}
res = {}
for name, fn in methods.items():
    res[name] = evaluate(fn)
    print(f"{name:40s} F1@0.5 {res[name].f1.mean():.4f} | F1@fold안 임계값 {res[name].f1_thr.mean():.4f} | AUC {res[name].auc.mean():.4f}", flush=True)

base = res[f"단일: {BEST}"]["f1"].values
n_tr, n_va = 5120, 1280
def ptest(d):
    v = d.var(ddof=1); t = d.mean() / np.sqrt((1 / len(d) + n_va / n_tr) * v) if v > 0 else 0
    return 2 * stats.t.sf(abs(t), len(d) - 1)
rows = []
for name, r in res.items():
    for col, lab in (("f1", "임계값 0.5"), ("f1_thr", "fold 안 임계값")):
        d = r[col].values - base
        rows.append((name, lab, r[col].mean(), r[col].std(ddof=0), d.mean(), ptest(d) if name != f"단일: {BEST}" or col != "f1" else np.nan,
                     int((d > 0).sum()), r.thr.median() if col == "f1_thr" else 0.5))
summary = pd.DataFrame(rows, columns=["방법", "임계값", "F1", "F1_std", "기준 대비", "p값", "10 fold 중 개선", "임계값(중앙값)"])
summary.to_csv(RES / "summary.csv", index=False)
display(summary.round(4))
""")

md("## 4. 시각화")
code("""
long = pd.concat([r.assign(method=n) for n, r in res.items()])
long = long.melt(id_vars=["method", "repeat", "fold"], value_vars=["f1", "f1_thr"], var_name="threshold", value_name="F1")
long["threshold"] = long.threshold.map({"f1": "0.5", "f1_thr": "in-fold tuned"})
long["method"] = long.method.str.replace("단일: ", "single ").str.replace("평균: 3차 튜닝 3개", "mean of 3 tuned") \\
    .str.replace("스태킹: 3차 튜닝 3개", "stack 3 tuned").str.replace("스태킹: 전체 후보", "stack all").str.replace("탐욕적 선택: 상위 12개 후보", "greedy top12")
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.boxplot(data=long, x="method", y="F1", hue="threshold", ax=ax, palette=["#cfd8e3", "#2b5d9a"])
ax.set_xlabel(""); ax.set_title("F1 by ensemble method (10 folds, all choices made inside folds)"); ax.tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(RES / "fig_methods.png", dpi=130); plt.show()
""")
code("""
# 스태킹(전체 후보)의 가중치: 개발 세트 전체 OOF로 한 번 학습해 해석용으로 본다
names = list(pool)
Z = np.column_stack([logit(np.mean([pool[n]["r0"], pool[n]["r1"]], 0)) for n in names])
lr = LogisticRegression(C=0.1, max_iter=2000).fit(Z, yd)
w = pd.Series(lr.coef_[0], index=names).sort_values()
w.to_csv(RES / "stack_weights.csv")
fig, ax = plt.subplots(figsize=(8, max(4, len(w) * 0.22)))
w.rename(lambda n: n.replace("1차:", "S1 ").replace("2차:", "S2 ").replace("3차:", "S3 ")).plot.barh(ax=ax, color=["#e3893f" if v < 0 else "#2b5d9a" for v in w]); ax.set_title("Stacking weights (logit inputs, C=0.1)")
plt.tight_layout(); plt.savefig(RES / "fig_stack_weights.png", dpi=130); plt.show()

g = m_greedy(top_prob); g("r0", np.ones(len(yd), bool), np.ones(len(yd), bool))
print("탐욕적 선택(개발 세트 전체, 반복0) 구성:", pd.Series(g.last).value_counts().to_dict())
""")

md("""
## 5. 최종 방법 확정 (제출 파일에 사용)
- 3절 표에서 F1이 가장 높은 (방법, 임계값 방식)을 고른다.
- 구성(모델·가중치·임계값)은 개발 세트 전체의 OOF(반복 0·1 평균)로 다시 정해 `results/final_method.json`에 저장한다.
""")
code("""
best_row = summary.sort_values("F1", ascending=False).iloc[0]
METHOD, THR_MODE = best_row["방법"], best_row["임계값"]
p_avg = {n: np.mean([pool[n]["r0"], pool[n]["r1"]], 0) for n in pool}
if METHOD.startswith("단일"):
    kind, names, weights, C = "mean", [BEST], [1.0], None
elif METHOD.startswith("평균"):
    kind, names, weights, C = "mean", tuned, [1 / len(tuned)] * len(tuned), None
elif METHOD.startswith("탐욕"):
    g = m_greedy(top_prob)
    for n in top_prob: pool[n]["avg"] = p_avg[n]
    g("avg", np.ones(len(yd), bool), np.ones(len(yd), bool))
    vc_ = pd.Series(g.last).value_counts(); kind, names, weights, C = "mean", list(vc_.index), list(vc_.values / vc_.sum()), None
else:
    names = tuned if "3개" in METHOD else list(pool); C = 1.0 if "3개" in METHOD else 0.1; kind, weights = "stack", None
if kind == "mean":
    p_full = np.sum([w * p_avg[n] for n, w in zip(names, weights)], 0)
else:
    Zs = np.vstack([np.column_stack([logit(pool[n][r]) for n in names]) for r in ("r0", "r1")])
    lr_f = LogisticRegression(C=C, max_iter=2000).fit(Zs, np.concatenate([yd, yd]))
    p_full = lr_f.predict_proba(np.column_stack([logit(p_avg[n]) for n in names]))[:, 1]
    weights = list(lr_f.coef_[0]); json_intercept = float(lr_f.intercept_[0])
thr = float(best_thr(yd, p_full)) if THR_MODE == "fold 안 임계값" else 0.5
final = {"method": METHOD, "threshold_mode": THR_MODE, "kind": kind, "models": names,
         "weights": [float(w) for w in weights], "C": C, "threshold": thr, "cv_f1": float(best_row["F1"])}
if kind == "stack": final["intercept"] = json_intercept
json.dump(final, open(RES / "final_method.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in final.items() if k not in ("weights",)}, ensure_ascii=False, indent=1))
""")

md("## 6. 결론과 마지막 차수로 넘길 것")
r_ = open("README.md", encoding="utf-8").read()
md(r_[r_.index("## 판단"):r_.index("## 파일")].replace("\n## ", "\n### ").replace("## 판단", "### 판단"))

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "exp04_threshold_ensemble.ipynb")
print("cells:", len(C))
