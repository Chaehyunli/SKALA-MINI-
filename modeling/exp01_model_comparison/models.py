"""1차 실험에서 비교하는 모델 정의 (하이퍼파라미터 고정, 튜닝은 3차)"""
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

SEED = 0
N_JOBS = 2


def build_models(cont_cols, fp_cols):
    """
    cont_cols: 연속·개수형 컬럼 (선형·SVM·MLP에서만 표준화, fold 안에서 fit)
    fp_cols:   Fingerprint 비트 컬럼 (KNN은 이것만 Jaccard 거리로 사용)
    """
    def scaled(est):
        pre = ColumnTransformer([("sc", StandardScaler(), cont_cols)], remainder="passthrough")
        return make_pipeline(pre, est)

    def fp_only(est):
        pre = ColumnTransformer([("fp", "passthrough", fp_cols)], remainder="drop")
        return make_pipeline(pre, FunctionTransformer(lambda a: a.astype(bool)), est)

    return {
        "LogReg": lambda: scaled(LogisticRegression(C=0.05, max_iter=3000)),
        "SVM_RBF": lambda: scaled(SVC(C=1.0, kernel="rbf", gamma="scale", random_state=SEED)),
        "KNN_Jaccard": lambda: fp_only(KNeighborsClassifier(n_neighbors=15, weights="distance",
                                                            metric="jaccard", n_jobs=N_JOBS)),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                                       n_jobs=N_JOBS, random_state=SEED),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                                                   n_jobs=N_JOBS, random_state=SEED),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                               colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                               verbose=-1, random_state=SEED, n_jobs=N_JOBS),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                             subsample=0.8, colsample_bytree=0.3, tree_method="hist",
                                             eval_metric="logloss", random_state=SEED, n_jobs=N_JOBS),
        "CatBoost": lambda: CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                               rsm=0.3, random_seed=SEED, thread_count=N_JOBS, verbose=0,
                                               allow_writing_files=False),
        "MLP": lambda: scaled(MLPClassifier((128,), alpha=1e-3, early_stopping=True,
                                            max_iter=200, random_state=SEED)),
    }
