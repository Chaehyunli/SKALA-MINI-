# Day2 모델링 실험 요약 (1~4차)

## 공통 설정
- 데이터: train.csv 8,349행 → 정제(라벨 충돌 parent 제외, parent당 1행) → 8,000행
- 분할(`common/splits.csv`, 고정): parent 그룹 기준. 테스트 20%(1,600행), 개발 80%(6,400행)를 5-fold × 2회 = 10 fold로 검증
- 평가: F1 (1 = 독성 없음 Positive). 비교는 fold별 짝지은 보정 t-검정. 테스트 세트는 차수별 최고 모델 확인에만 사용
- 제출 형식: 교수님 예시와 같은 `SMILES, output` 927행 (`common/submission.py`가 행 수·순서·값을 검사)

## 차수별 결과
| 차수 | 한 일 | 차수 최고 | CV F1 | 테스트 F1 | 제출 파일 |
|---|---|---|---|---|---|
| 1차 | 9개 모델 비교 (입력 1,042컬럼 고정) | RandomForest | 0.8177 | 0.813 | `exp01_model_comparison/predict_output_임채현_1차.csv` |
| 2차 | 입력 표현 변경 (RDKit 217, Mordred, PTFP, MACCS, AP/TT, 입체 count FP, SVD) | 입체 count FP × RandomForest | 0.8191 | 0.821 | `exp02_feature_variants/predict_output_임채현_2차.csv` |
| 3차 | Optuna 튜닝 (+MACCS 입력, 상위 3개 모델) | 튜닝 XGBoost | 0.8218 | 0.809 | `exp03_tuning/predict_output_임채현_3차.csv` |
| 4차 | 33개 모델 스태킹 + 임계값 조정 | 스태킹 + 임계값 0.43 | 0.8282 | 0.822 | `exp04_threshold_ensemble/predict_output_임채현_4차.csv` |

## 핵심 정리
- 1차: 트리 계열이 1~5위를 차지했다. 상위 3개(RF·LightGBM·XGBoost)끼리는 차이가 없다.
- 2차: 입력 표현을 바꾸는 것만으로는 의미 있는 상승이 없었다. descriptor 추가(RDKit·Mordred)는 오히려 조금 낮아졌다. 대신 입력이 다른 모델들은 앙상블 재료로 쓸모가 있었다.
- 3차: 튜닝으로 세 모델이 0.821~0.822가 됐다. XGBoost만 유의하게 개선됐지만, 테스트에서는 낮게 나와 단일 모델의 불안정성이 보였다.
- 4차: 여러 입력·여러 모델을 스태킹하고 임계값을 0.43으로 낮춘 것이 CV와 테스트 모두에서 가장 높았다. 최종 제출 후보로 가장 적합하다.
- 한계: 구조가 거의 같은 화합물(유사도 0.9 이상)끼리도 라벨 일치율이 0.81이다. 현재 특성으로는 F1 0.83 근처가 상한에 가깝다. 더 올리려면 GNN이나 사전학습 분자 언어모델처럼 SMILES 구조를 직접 학습하는 모델을 추가해야 한다.

## 실행 순서 (맥에서 다시 돌릴 때)
1. `pip install -r ../requirements.txt` + `pip install mordredcommunity optuna pyarrow`
2. `common/make_splits.py` (splits.csv가 이미 있으면 생략)
3. 1차 → 2차(`build_features.py` 후 노트북; cache가 있으면 생략 가능) → 3차 → 4차 순서로 노트북 실행
4. 각 폴더의 `predict_submission.ipynb`로 그 차수 제출 파일 생성
