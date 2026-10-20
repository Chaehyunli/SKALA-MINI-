# 1차 실험: 모델 계열 비교 (입력 1,042컬럼 고정)

## 목적
Day1 EDA에서 확정한 1,042컬럼을 그대로 두고, 9개 모델을 같은 조건으로 비교한다. 2·3차에서 집중할 모델과 성능을 올릴 방향을 정하는 것이 목표다.

## 설정
| 항목 | 내용 |
|---|---|
| 입력 | 이 폴더의 `preprocess.py` + `feature_columns.json` (Day1 버전) → 8,000행 × 1,042컬럼 (ECFP 644 + FCFP 383 + Descriptor 3 + RDKit 12) |
| 분할 | `../common/splits.csv` — parent 그룹 기준. 테스트 20%(1,600행)는 사용 안 함. 개발 80%(6,400행)를 5-fold × 2회 = 10 fold로 검증 |
| 지표 | F1 (임계값 0.5), 보조로 AUC·Precision·Recall·학습 시간 |
| 하이퍼파라미터 | 모델별 기본값 고정 (`models.py`), 튜닝은 3차 |
| 비교 방법 | 평균 ± 표준편차 + 1위 모델과의 보정 짝지은 t-검정 (Nadeau & Bengio, 2003) |

## 실행 방법
1. 처음 한 번: `pip install xgboost catboost` (requirements.txt에 추가됨)
2. `../common/splits.csv`가 없으면 `../common`에서 `python make_splits.py` 실행
3. `exp01_model_comparison.ipynb`를 위에서부터 실행 (약 8~10분). `RUN_CV = False`로 두면 `results/`의 저장 결과만 불러온다.

## 결과
| 순위 | 모델 | F1 | F1 std | AUC | Precision | Recall | 1위와 차이 (p) | 학습 시간/fold |
|---|---|---|---|---|---|---|---|---|
| 1 | RandomForest | 0.8177 | 0.0063 | 0.8737 | 0.794 | 0.843 | — | 3.9s |
| 2 | LightGBM | 0.8150 | 0.0062 | 0.8757 | 0.802 | 0.829 | −0.003 (0.44) | 2.2s |
| 3 | XGBoost | 0.8131 | 0.0100 | 0.8729 | 0.797 | 0.830 | −0.005 (0.41) | 2.5s |
| 4 | CatBoost | 0.8097 | 0.0067 | 0.8683 | 0.791 | 0.829 | −0.008 (0.048) | 12.8s |
| 5 | ExtraTrees | 0.8086 | 0.0064 | 0.8614 | 0.793 | 0.825 | −0.009 (0.030) | 5.7s |
| 6 | SVM_RBF | 0.8040 | 0.0048 | 0.8631 | 0.783 | 0.827 | −0.014 (0.003) | 6.4s |
| 7 | MLP | 0.7962 | 0.0075 | 0.8530 | 0.778 | 0.817 | −0.021 (0.003) | 1.6s |
| 8 | LogReg | 0.7798 | 0.0044 | 0.8315 | 0.764 | 0.796 | −0.038 (<0.001) | 0.8s |
| 9 | KNN_Jaccard | 0.7652 | 0.0094 | 0.8240 | 0.766 | 0.765 | −0.053 (<0.001) | 0.04s |

- 모두 1로 예측할 때의 F1은 0.703이다. 9개 모델 모두 이보다 높다.
- 학습 데이터가 6,400행의 4/5(5,120행)라서 Day1(8,000행의 4/5)보다 점수가 0.003 정도 낮다. Day1 LightGBM 0.818 → 이번 0.815.

## 판단
1. 상위 그룹은 RandomForest · LightGBM · XGBoost다. 1위 RandomForest와 LightGBM·XGBoost의 차이는 통계적으로 구분되지 않는다(p = 0.44, 0.41). 반대로 CatBoost 이하 6개는 1위보다 유의하게 낮다.
2. 트리 계열이 1~5위를 모두 차지했다. 비트 1,027개와 연속형 15개가 섞인 입력에서, 스케일링 없이 변수 간 상호작용을 학습하는 트리가 유리했다.
3. 거리 기반 KNN이 가장 낮았다. Fingerprint 유사도만으로는 부족하고 물리·화학 특성이 함께 필요하다는 뜻이다.
4. KNN을 뺀 8개 모델에서 Recall이 Precision보다 높다(KNN은 둘이 같음). 비독성(1)으로 예측하는 경향이 있어서, 독성 물질을 안전하다고 판단하는 오류가 더 많다.

## 인사이트 (2차 이후 방향의 근거)
| 관찰 | 수치 | 의미 |
|---|---|---|
| 연속형 특성의 기여가 크다 | LightGBM gain에서 Descriptor 3개가 13.1%(컬럼당 평균의 45배), RDKit 12개가 11.9%(10배). 상위 6개 특성이 모두 연속형(clogp, TPSA, MolWt, frac_csp3, sa_score, n_rotb) | 물리·화학 특성을 늘리면 성능이 오를 여지가 크다. RDKit 전체 descriptor(217개)나 Mordred(약 1,600개)를 시도할 근거 |
| (주의) gain의 한계 | 연속형 변수는 나눌 수 있는 지점이 많아 0/1 비트보다 gain이 크게 잡히는 경향이 있다 | 정확한 기여는 2차에서 descriptor를 실제로 늘려 F1로 확인한다 |
| 라벨 상관이 약했던 변수도 트리에서는 중요 | MolWt(라벨 상관 0.01 미만)가 gain 3위, frac_csp3가 4위 | 단독 상관으로 변수를 빼지 않은 Day1 판단이 맞았다. 다른 변수와의 상호작용으로 쓰인다 |
| 처음 보는 구조에서 정답률이 떨어진다 | 학습 데이터와 최대 유사도 0.8 이상: 정답률 0.839 / 0.4 미만: 0.742 / 0.4~0.5: 0.726 | 구조가 낯선 화합물을 보완할 특성(물리·화학 descriptor, 다른 종류의 Fingerprint)이 필요하다 |
| 임계값 조정 효과는 작다 | 최적 임계값 0.40~0.44, F1 상승 +0.002~+0.004 (트리 모델, OOF 기준이라 낙관적) | 4차에서 할 일이지만 큰 개선은 기대하기 어렵다 |
| 상위 트리 모델끼리 예측이 거의 같다 | 예측 순위 상관 0.94~0.98. RandomForest + LightGBM 평균 앙상블 0.8192(+0.0015) | 같은 입력의 트리끼리 섞는 앙상블은 효과가 작다. 다른 입력 표현으로 학습한 모델을 섞어야 한다 |

## 2차 실험 제안 (우선순위 순)
모델은 상위 3개(RandomForest, LightGBM, XGBoost)로 좁히고, 비교용으로 SVM_RBF를 함께 둔다.

| 우선순위 | 시도 | 근거 | 방법 |
|---|---|---|---|
| 1 | 물리·화학 descriptor 확장 | 연속형 특성의 컬럼당 기여가 Fingerprint의 10~45배 | RDKit 전체 217개 → 상수·중복(상관 0.95 이상) 제거. 이후 Mordred(약 1,600개)도 같은 방식으로 비교 |
| 2 | Fingerprint 표현 바꾸기 | 처음 보는 구조에서 정답률 0.74 | 입체 정보 반영(useChirality), 비트 대신 개수(count) Fingerprint, radius 3, 2,048비트. MACCS(166), Atom-pair, Topological torsion 추가 |
| 3 | PTFP 다시 추가 | Day1에서는 고정 하이퍼파라미터로 차이가 없었음 | 트리 모델에서만 재확인 |
| 4 | 차원 축소 · 특성 선택 | 컬럼이 수천 개로 늘어날 때 필요 | Fingerprint에 TruncatedSVD(100~300차원), fold 안에서 중요도 상위 k개 선택. 트리보다 SVM·MLP·LogReg에서 효과를 기대 |

- 1·2번을 하면 `preprocess.py`가 바뀌므로, 2차 폴더에 새 `preprocess.py`를 둔다. 분할은 `common/splits.csv`를 그대로 써서 1차와 같은 fold로 비교한다.
- 변수 선택이나 차원 축소는 반드시 fold 안에서 fit한다. 개발 세트 전체로 고르면 검증 점수가 부풀려진다.

## 3차 이후 (예정)
- 3차: 상위 모델 하이퍼파라미터 튜닝 (Optuna, fold 안에서). RandomForest는 `max_features`·`min_samples_leaf`, 부스팅은 `num_leaves`·학습률·트리 수·정규화
- 4차: 임계값 최적화(fold 안에서 선택) + 서로 다른 입력으로 학습한 모델의 앙상블·스태킹
- 5차: 테스트 세트 1회 평가 → 8,000행 전체로 재학습 → `predict_input.csv` 927행 예측 → `predict_output_작성자.csv` 생성

## 파일
| 파일 | 내용 |
|---|---|
| `exp01_model_comparison.ipynb` | 실험 노트북 (실행 결과 포함) |
| `models.py` | 9개 모델 정의 |
| `preprocess.py`, `feature_columns.json` | Day1 전처리 (복사본) |
| `build_nb.py` | 노트북 생성 스크립트 |
| `results/cv_scores.csv` | 모델 × fold별 F1·AUC·Precision·Recall·학습 시간 |
| `results/oof_predictions.csv` | 개발 세트 6,400행의 OOF 예측 확률 (모델 × 반복) — 4차 앙상블에 재사용 |
| `results/summary.csv`, `paired_ttest_vs_best.csv` | 요약표, 짝지은 검정 결과 |
| `results/fig_*.png` | 그래프 8개 |
