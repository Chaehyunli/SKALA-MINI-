"""
마감 대응 5차 제출: 8차 최종 스태킹 확률 + GNN 확률을 2단계 스태킹으로 결합 (임계값 0.5)

1) 8차 스태킹(41개)의 OOF 확률을 fold 안에서 다시 만든다 (fold k는 나머지 fold로 학습한 스태킹으로 예측)
2) [8차 스태킹 logit, GNN logit] 두 입력의 로지스틱 회귀를 fold 안에서 평가 → 8차 단독과 짝지은 비교
3) 개발 세트 전체 OOF로 2단계 계수를 정하고, 8,000행 전체로 학습한 8차 확률·GNN 확률에 적용
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent; M = HERE.parent
sys.path.insert(0, str(M / "common"))
from preprocess import preprocess_train
from evaluate import load_splits
from submission import write_submission
import stacking as S

df, X, y, g = preprocess_train(M.parent / "train.csv"); df = load_splits(df)
d = df[df.set == "dev"].reset_index(drop=True); yd = d.label.values; smi = d.SMILES.values
folds = d[["fold_r0", "fold_r1"]]
pool = S.full_pool(smi, "없음")
f8 = json.load(open(M / "exp08_pseudo_label/results/final_method.json")); names = f8["models"]

oof8 = {}
for ri, r in enumerate(("r0", "r1")):
    f = folds[f"fold_r{ri}"].values; Z = np.column_stack([S.logit(pool[n][r]) for n in names]); o = np.zeros(len(yd))
    for k in range(5):
        tr, va = f != k, f == k
        o[va] = LogisticRegression(C=0.1, max_iter=3000).fit(Z[tr], yd[tr]).predict_proba(Z[va])[:, 1]
    oof8[r] = o
pd.DataFrame({"SMILES": smi, "r0": oof8["r0"], "r1": oof8["r1"]}).to_csv(HERE / "results_fast" / "oof_stack8.csv", index=False)

gfile = HERE / "results" / "oof_predictions.csv"
if not gfile.exists():
    print("GNN OOF 아직 없음 — 8차 스태킹 OOF만 저장"); sys.exit(0)
go = pd.read_csv(gfile); assert (go.SMILES.values == smi).all()
gnn = {"r0": go["gnn|r0"].values, "r1": go["gnn|r1"].values}

rows = []
for ri, r in enumerate(("r0", "r1")):
    f = folds[f"fold_r{ri}"].values
    Z2 = np.column_stack([S.logit(oof8[r]), S.logit(gnn[r])])
    for k in range(5):
        tr, va = f != k, f == k
        p2 = LogisticRegression(C=1.0, max_iter=3000).fit(Z2[tr], yd[tr]).predict_proba(Z2[va])[:, 1]
        rows.append(dict(repeat=ri, fold=k, f1_8=f1_score(yd[va], oof8[r][va] >= .5), f1_5=f1_score(yd[va], p2 >= .5),
                         auc_8=roc_auc_score(yd[va], oof8[r][va]), auc_5=roc_auc_score(yd[va], p2)))
R = pd.DataFrame(rows); dd = R.f1_5.values - R.f1_8.values
print(f"F1@0.5  8차 스태킹 {R.f1_8.mean():.4f} → +GNN {R.f1_5.mean():.4f} ({dd.mean():+.4f}, p={S.paired_p(dd):.3f}, 10 fold 중 개선 {(dd > 0).sum()})")
print(f"AUC     8차 스태킹 {R.auc_8.mean():.4f} → +GNN {R.auc_5.mean():.4f}")
print(f"GNN 단독 F1@0.5 {np.mean([f1_score(yd, gnn[r] >= .5) for r in ('r0', 'r1')]):.4f}")
R.to_csv(HERE / "results_fast" / "two_level_cv.csv", index=False)

Zs = np.vstack([np.column_stack([S.logit(oof8[r]), S.logit(gnn[r])]) for r in ("r0", "r1")])
lr = LogisticRegression(C=1.0, max_iter=3000).fit(Zs, np.concatenate([yd, yd]))
print("2단계 계수 [8차, GNN]:", np.round(lr.coef_[0], 3), "절편", round(float(lr.intercept_[0]), 3))
gf = HERE / "results_fast" / "gnn_predict_full.csv"
if not gf.exists():
    print("전체 데이터 GNN 예측이 아직 없음"); sys.exit(0)
p8 = pd.read_csv(M / "exp08_pseudo_label/results/predict_proba.csv"); pg = pd.read_csv(gf)
assert (p8.SMILES.values == pg.SMILES.values).all()
p = lr.predict_proba(np.column_stack([S.logit(p8.score_nontoxic.values), S.logit(pg.gnn_prob.values)]))[:, 1]
path, sub = write_submission(p8.SMILES, (p >= 0.5).astype(int), HERE, "5차_thr0.5")
prev = pd.read_csv(M / "exp08_pseudo_label/predict_output_임채현_8차_thr0.5.csv").iloc[:, 1].values
v = sub.iloc[:, 1].values
print("저장:", path.name, "| 비독성 비율", round(v.mean(), 3), f"| 8차_thr0.5와 다른 행 {int((v != prev).sum())} (5차만 1: {int(((v==1)&(prev==0)).sum())}, 8차만 1: {int(((v==0)&(prev==1)).sum())})")
json.dump({"method": "2단계 스태킹: 8차 최종 확률 + GNN", "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "threshold": 0.5,
           "cv_f1_8": float(R.f1_8.mean()), "cv_f1_5": float(R.f1_5.mean()), "p": float(S.paired_p(dd))},
          open(HERE / "results_fast" / "final_fast.json", "w"), ensure_ascii=False, indent=1)
