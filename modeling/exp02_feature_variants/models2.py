"""2차 실험 모델 정의 (exp02 노트북과 같은 설정) — 제출 파일 만들 때 사용"""
import lightgbm as lgb
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED, N_JOBS = 0, 2

ALGOS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=N_JOBS, random_state=SEED),
    "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3,
                                           subsample=0.8, subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=N_JOBS),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                         colsample_bytree=0.3, tree_method="hist", eval_metric="logloss",
                                         random_state=SEED, n_jobs=N_JOBS),
}


def svd_algos(fp_cols, other_cols):
    def pipe(est):
        pre = ColumnTransformer([("svd", TruncatedSVD(256, random_state=SEED), fp_cols),
                                 ("sc", StandardScaler(), other_cols)])
        return make_pipeline(pre, est)
    return {
        "SVM_RBF": lambda: pipe(SVC(C=1.0, kernel="rbf", gamma="scale", random_state=SEED)),
        "MLP": lambda: pipe(MLPClassifier((128,), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED)),
        "LightGBM": lambda: pipe(ALGOS["LightGBM"]()),
    }
