"""
모든 차수에서 공통으로 쓰는 분할 로드 · 교차검증 · 지표 · 짝지은 검정 함수
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
SPLITS_FILE = HERE / "splits.csv"


def load_splits(df):
    """preprocess 결과 df(SMILES 포함)에 set, fold_r* 컬럼을 SMILES 기준으로 붙인다"""
    sp = pd.read_csv(SPLITS_FILE)
    out = df.merge(sp[["SMILES", "set"] + [c for c in sp.columns if c.startswith("fold_r")]],
                   on="SMILES", how="left", validate="one_to_one")
    if out["set"].isna().any():
        raise ValueError(f"splits.csv에 없는 SMILES {out['set'].isna().sum()}개 — make_splits.py를 다시 실행하세요")
    return out


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(f1=f1_score(y, pred), auc=roc_auc_score(y, p),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred))


def run_cv(make_model, X, y, fold_cols, name, fit_kwargs=None):
    """
    make_model(): 새 모델(또는 파이프라인)을 돌려주는 함수
    X: 개발 세트 특성(DataFrame), y: 라벨(ndarray), fold_cols: DataFrame[fold_r0, fold_r1]
    반환: (fold별 점수 DataFrame, OOF 확률 DataFrame[repeat별 컬럼])
    """
    rows, oof = [], {}
    for r, col in enumerate(fold_cols.columns):
        folds = fold_cols[col].values
        p_oof = np.zeros(len(y))
        for k in sorted(np.unique(folds)):
            tr, va = folds != k, folds == k
            m = make_model()
            t0 = time.time()
            m.fit(X[tr], y[tr], **(fit_kwargs or {}))
            fit_t = time.time() - t0
            p = predict_score(m, X[va])
            p_oof[va] = p
            rows.append(dict(model=name, repeat=r, fold=k, fit_sec=fit_t, n_train=int(tr.sum()),
                             n_val=int(va.sum()), **metrics(y[va], p)))
        oof[f"r{r}"] = p_oof
    return pd.DataFrame(rows), pd.DataFrame(oof)


def predict_score(m, X):
    """predict_proba가 없으면(SVM) decision_function을 시그모이드로 0~1로 변환 (0.5 = 결정경계)"""
    if hasattr(m, "predict_proba"):
        try:
            return m.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    d = m.decision_function(X)
    return 1 / (1 + np.exp(-d))


def summarize(scores):
    g = scores.groupby("model")
    tbl = pd.DataFrame({
        "F1": g.f1.mean(), "F1_std": g.f1.std(ddof=0),
        "AUC": g.auc.mean(), "AUC_std": g.auc.std(ddof=0),
        "Precision": g.precision.mean(), "Recall": g.recall.mean(),
        "fit_sec(1fold)": g.fit_sec.mean(),
    })
    return tbl.sort_values("F1", ascending=False)


def corrected_paired_ttest(scores, base, metric="f1"):
    """
    Nadeau & Bengio (2003) 보정 짝지은 t-검정.
    같은 (repeat, fold)에서 나온 두 모델 점수의 차이를 비교한다.
    CV fold들은 학습 데이터가 겹쳐 독립이 아니므로 분산에 (1/J + n_val/n_train)을 곱해 보정한다.
    """
    piv = scores.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    n_tr = scores.n_train.mean(); n_va = scores.n_val.mean()
    J = len(piv)
    out = []
    for m in piv.columns:
        if m == base:
            continue
        d = (piv[m] - piv[base]).values
        var = d.var(ddof=1)
        t = d.mean() / np.sqrt((1 / J + n_va / n_tr) * var) if var > 0 else 0.0
        p = 2 * stats.t.sf(abs(t), df=J - 1)
        out.append(dict(model=m, diff_vs_best=d.mean(), t=t, p_value=p, wins=int((d > 0).sum()), folds=J))
    return pd.DataFrame(out).set_index("model").sort_values("diff_vs_best", ascending=False)
