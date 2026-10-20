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


# ---------- 6차 이후 공통: 이전 차수 OOF 모으기와 "4차 + 새 모델" 비교 ----------
from pathlib import Path as _P
_M = _P(__file__).resolve().parent.parent


def base_pool(smi):
    """4차 최종 스태킹에 들어간 33개 모델의 OOF (1~3차)"""
    pool = {}
    pool.update(read_oof(_M / "exp01_model_comparison/results/oof_predictions.csv", "1차:", smi))
    pool.update({k: v for k, v in read_oof(_M / "exp02_feature_variants/results/oof_predictions.csv", "2차:", smi).items()
                 if not k.startswith("2차:base|")})
    pool.update(read_oof(_M / "exp03_tuning/results/oof_predictions.csv", "3차:", smi))
    return {k: v for k, v in pool.items() if set(v) == {"r0", "r1"}}


def compare_with_s4(pool, base_names, new_names, y, folds, extra_methods=None):
    """4차 최종(스태킹 33개, C=0.1, fold 안 임계값)과 '33개 + 새 모델' 등을 같은 fold로 비교"""
    methods = {"4차 최종: 스태킹 33개": m_stack(pool, base_names, y, C=0.1),
               "스태킹 33개 + 새 모델": m_stack(pool, base_names + new_names, y, C=0.1)}
    methods.update(extra_methods or {})
    res = {n: evaluate(fn, y, folds) for n, fn in methods.items()}
    ref = res["4차 최종: 스태킹 33개"]["f1_thr"].values
    rows = []
    for n, r in res.items():
        dd = r["f1_thr"].values - ref
        rows.append((n, r["f1"].mean(), r["f1_thr"].mean(), r["f1_thr"].std(ddof=0), r.auc.mean(), dd.mean(),
                     paired_p(dd), int((dd > 0).sum()), r.thr.median()))
    tbl = pd.DataFrame(rows, columns=["방법", "F1@0.5", "F1(fold 안 임계값)", "F1_std", "AUC", "4차 최종 대비", "p값",
                                      "10 fold 중 개선", "임계값(중앙값)"])
    return tbl, res


def finalize(pool, names, y, C, path, method, cv_f1, extra=None):
    """개발 세트 전체 OOF(반복 0·1)로 스태킹 계수와 임계값을 다시 정해 저장"""
    import json
    Zs = np.vstack([np.column_stack([logit(pool[n][r]) for n in names]) for r in ("r0", "r1")])
    lr = LogisticRegression(C=C, max_iter=3000).fit(Zs, np.concatenate([y, y]))
    p_full = lr.predict_proba(np.column_stack([logit(np.mean([pool[n]["r0"], pool[n]["r1"]], 0)) for n in names]))[:, 1]
    final = {"method": method, "kind": "stack", "models": names, "weights": [float(w) for w in lr.coef_[0]], "C": C,
             "intercept": float(lr.intercept_[0]), "threshold": float(best_thr(y, p_full)), "cv_f1": float(cv_f1)}
    final.update(extra or {})
    json.dump(final, open(path, "w"), ensure_ascii=False, indent=1)
    return final


STAGE_DIRS = [("4차", "exp04_threshold_ensemble"), ("6차", "exp06_knn_features"), ("7차", "exp07_seed_avg"),
              ("8차", "exp08_pseudo_label"), ("9차", "exp09_mordred_retry")]


def full_pool(smi, upto):
    """1~3차 + 6차 ~ (upto 직전 차수)까지의 OOF를 모두 모은다"""
    pool = base_pool(smi)
    for st, d in STAGE_DIRS[1:]:
        if st == upto: break
        f = _M / d / "results" / "oof_predictions.csv"
        if f.exists(): pool.update(read_oof(f, f"{st}:", smi))
    return pool


def best_so_far(upto):
    """upto 직전 차수까지의 최종 방법 중 CV F1이 가장 높은 것 (라벨, 모델 목록, C, cv_f1)"""
    import json
    best = None
    for st, d in STAGE_DIRS:
        if st == upto: break
        f = _M / d / "results" / "final_method.json"
        if not f.exists(): continue
        j = json.load(open(f))
        if j.get("kind", "stack") != "stack": continue
        if best is None or j["cv_f1"] > best[3]:
            best = (f"지금까지 최고: {st} 최종", j["models"], j.get("C") or 0.1, j["cv_f1"])
    return best
