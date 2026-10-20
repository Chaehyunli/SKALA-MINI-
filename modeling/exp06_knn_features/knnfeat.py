"""
6차: 이웃 라벨 특성 (k-nearest-neighbor label features)

학습 데이터 중 구조가 가장 비슷한 화합물들의 라벨을 특성으로 만든다.
- 유사도: 제공된 ECFP·FCFP 1,024비트 전체의 Tanimoto 유사도
- 특성 (Fingerprint 종류마다 8개, 총 16개)
  max_sim, mean_sim_top5, wlab_k1/k3/k5/k10 (유사도 가중 라벨 평균), lab_ge07 (유사도 0.7 이상 이웃의 라벨 평균), n_ge07
- 누수 방지: 이 특성은 반드시 "학습에 쓰는 데이터"로만 만든다.
  · 학습 행: 같은 학습 데이터 안에서 자기 자신을 뺀 이웃으로 계산 (leave-one-out)
  · 검증·테스트·예측 행: 학습 데이터 전체를 이웃 후보로 계산
  KnnAugmented가 fit에서 학습 데이터를 기억하고, predict에서 그 데이터만 이웃으로 쓴다.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

FP_PREFIX = {"E": "fullecfp_", "F": "fullfcfp_"}
KS = (1, 3, 5, 10)


def full_fp(df):
    """preprocess 결과 df(원본 ecfp_/fcfp_ 1,024비트 포함)에서 이웃 계산용 전체 비트를 꺼낸다"""
    e = df[[f"ecfp_{i}" for i in range(1024)]].to_numpy(np.float32)
    f = df[[f"fcfp_{i}" for i in range(1024)]].to_numpy(np.float32)
    cols = [f"fullecfp_{i}" for i in range(1024)] + [f"fullfcfp_{i}" for i in range(1024)]
    return pd.DataFrame(np.hstack([e, f]), columns=cols, index=df.index)


def _tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def _feats(S, y_ref, prior, tag):
    order = np.argsort(-S, axis=1)[:, :max(KS)]
    s = np.take_along_axis(S, order, 1); lab = y_ref[order]
    out = {f"{tag}_max_sim": s[:, 0], f"{tag}_mean_sim_top5": s[:, :5].mean(1)}
    for k in KS:
        w = s[:, :k] + 1e-6
        out[f"{tag}_wlab_k{k}"] = (w * lab[:, :k]).sum(1) / w.sum(1)
    m = S >= 0.7
    n = m.sum(1)
    out[f"{tag}_lab_ge07"] = np.where(n > 0, (m * y_ref[None, :]).sum(1) / np.maximum(n, 1), prior)
    out[f"{tag}_n_ge07"] = n.astype(np.float32)
    return out


def knn_features(fp_ref, y_ref, fp_q, self_exclude=False, chunk=2000):
    """fp_ref/fp_q: DataFrame(full ecfp+fcfp). self_exclude=True면 fp_q가 fp_ref와 같은 행(자기 자신 제외)"""
    y_ref = np.asarray(y_ref, dtype=np.float32); prior = float(y_ref.mean())
    res = []
    for s in range(0, len(fp_q), chunk):
        part = {}
        for tag, pre in FP_PREFIX.items():
            cols = [c for c in fp_ref.columns if c.startswith(pre)]
            A = fp_q[cols].to_numpy(np.float32)[s:s + chunk]; B = fp_ref[cols].to_numpy(np.float32)
            S = _tanimoto(A, B)
            if self_exclude:
                S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -1
            part.update(_feats(S, y_ref, prior, tag))
        res.append(pd.DataFrame(part))
    return pd.concat(res, ignore_index=True).astype(np.float32)


class KnnAugmented(BaseEstimator, ClassifierMixin):
    """X의 full Fingerprint 컬럼으로 이웃 라벨 특성을 만들어 나머지 컬럼에 붙인 뒤 estimator를 학습"""

    def __init__(self, estimator, keep_original=True):
        self.estimator = estimator
        self.keep_original = keep_original

    def _split(self, X):
        fp = [c for c in X.columns if c.startswith(tuple(FP_PREFIX.values()))]
        return X[fp], X.drop(columns=fp)

    def fit(self, X, y):
        fp, rest = self._split(X)
        self.fp_ref_, self.y_ref_ = fp.reset_index(drop=True), np.asarray(y)
        K = knn_features(self.fp_ref_, self.y_ref_, self.fp_ref_, self_exclude=True)
        Z = pd.concat([rest.reset_index(drop=True), K], axis=1) if self.keep_original else K
        self.model_ = clone(self.estimator).fit(Z, y)
        self.classes_ = self.model_.classes_
        return self

    def _z(self, X):
        fp, rest = self._split(X)
        K = knn_features(self.fp_ref_, self.y_ref_, fp.reset_index(drop=True))
        return pd.concat([rest.reset_index(drop=True), K], axis=1) if self.keep_original else K

    def predict_proba(self, X):
        return self.model_.predict_proba(self._z(X))

    def predict(self, X):
        return self.model_.predict(self._z(X))
