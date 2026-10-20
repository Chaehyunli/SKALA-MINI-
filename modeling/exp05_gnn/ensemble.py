"""4차와 같은 방식(fold 안에서 규칙 결정)으로 앙상블을 평가하는 함수 — 5차에서 GNN을 추가해 다시 평가할 때 사용"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

THS = np.round(np.arange(0.30, 0.701, 0.01), 2)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def best_thr(y, p):
    return THS[int(np.argmax([f1_score(y, p >= t) for t in THS]))]


def read_oof(path, prefix, smi):
    o = pd.read_csv(path); assert (o["SMILES"].values == smi).all()
    out = {}
    for c in o.columns:
        if "|r" in c:
            name, r = c.rsplit("|", 1); out.setdefault(prefix + name, {})[r] = o[c].values
    return out


def evaluate(method, y, folds):
    rows = []
    for ri, r in enumerate(("r0", "r1")):
        f = folds[f"fold_r{ri}"].values
        for k in range(5):
            tr, va = f != k, f == k
            p_va, p_tr = method(r, tr, va)
            t = best_thr(y[tr], p_tr)
            rows.append(dict(repeat=ri, fold=k, f1=f1_score(y[va], p_va >= .5), f1_thr=f1_score(y[va], p_va >= t),
                             thr=t, auc=roc_auc_score(y[va], p_va)))
    return pd.DataFrame(rows)


def m_single(pool, name):
    return lambda r, tr, va: (pool[name][r][va], pool[name][r][tr])


def m_mean(pool, names):
    return lambda r, tr, va: (np.mean([pool[n][r][va] for n in names], 0), np.mean([pool[n][r][tr] for n in names], 0))


def m_stack(pool, names, y, C=1.0):
    def f(r, tr, va):
        Z = np.column_stack([logit(pool[n][r]) for n in names])
        lr = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], y[tr])
        return lr.predict_proba(Z[va])[:, 1], lr.predict_proba(Z[tr])[:, 1]
    return f


def paired_p(d, n_tr=5120, n_va=1280):
    v = d.var(ddof=1)
    if v == 0: return np.nan
    t = d.mean() / np.sqrt((1 / len(d) + n_va / n_tr) * v)
    return 2 * stats.t.sf(abs(t), len(d) - 1)
