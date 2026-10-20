# 2차 실험: 입력 표현 바꾸기 (특성 블록 추가 · 교체 · 차원 축소)

## 목적
1차 상위 모델(RandomForest, LightGBM, XGBoost)을 고정하고 입력 표현만 바꿔서 F1이 오르는지 확인한다. 1차 인사이트에서 나온 가설 두 가지를 검증한다.
- 연속형 descriptor를 늘리면 오른다.
- 처음 보는 구조에 강한 표현(다른 Fingerprint, 입체 정보)을 쓰면 오른다.

## 설정
| 항목 | 내용 |
|---|---|
| 기준(base) | Day1 1,042컬럼. 1차 결과를 그대로 재사용 (같은 fold, 같은 하이퍼파라미터) |
| 추가 블록 | `featurize.py`로 SMILES에서 계산, `build_features.py`가 `cache/`에 저장 (Mordred 계산 약 12분) |
| 블록 정리 | 라벨 미사용, 개발 세트로만 계산. 연속형은 결측 20% 초과·상수·기존 연속형과 \|상관\| ≥ 0.95 제거. Fingerprint형은 0이 아닌 비율 2~98%, 완전 중복 제거. 정리 기준은 `results/block_spec.json`에 저장해 예측 데이터에 그대로 적용 |
| 분할 · 평가 | 1차와 같은 `common/splits.csv` 10 fold. 모델별로 base와 fold별 짝지은 보정 t-검정 |

| 블록 | 원래 컬럼 | 정리 후 |
|---|---|---|
| RDKit 전체 descriptor | 217 | 168 |
| Mordred 2D descriptor | 1,613 | 700 |
| PTFP | 1,024 | 751 |
| MACCS | 167 | 125 |
| Atom-pair + Topological torsion (count) | 4,096 | 1,655 |
| ECFP4 count + 입체 정보 | 2,048 | 566 |
| FCFP4 count | 2,048 | 368 |

## 실행 방법
1. `python build_features.py` (한 번만, 약 15분)
2. `exp02_feature_variants.ipynb` 위에서부터 실행 (약 30분). `RUN_CV = False`면 저장 결과만 불러온다.
3. 제출 파일: `predict_submission.ipynb` 실행 → `predict_output_임채현_2차.csv`

## 결과 (F1, 10 fold 평균)
| 변형 | 컬럼 수 | RandomForest | LightGBM | XGBoost | 트리 3개 평균 |
|---|---|---|---|---|---|
| base (1차) | 1,042 | 0.8177 | 0.8150 | 0.8131 | 0.8152 |
| +maccs | 1,167 | 0.8190 | 0.8178 | 0.8142 | 0.8170 |
| +ptfp | 1,793 | 0.8155 | 0.8185 | 0.8153 | 0.8164 |
| +ap_tt | 2,697 | 0.8150 | 0.8180 | 0.8160 | 0.8164 |
| count_chiral | 949 | 0.8191 | 0.8143 | 0.8113 | 0.8149 |
| +rdkit217 | 1,210 | 0.8148 | 0.8122 | 0.8115 | 0.8128 |
| +mordred | 1,742 | 0.8106 | 0.8114 | 0.8165 | 0.8128 |

| 차원 축소 (SVD 256) | 모델 | F1 | base 대비 (p) |
|---|---|---|---|
| svd | LightGBM | 0.8000 | −0.0150 (0.002) |
| svd | MLP | 0.8008 | +0.0045 (0.34) |
| svd | SVM_RBF | 0.7983 | −0.0057 (0.057) |

- base 대비 차이는 모든 변형·트리 모델에서 −0.0071~+0.0035이고, p값이 모두 0.19 이상이다. 통계적으로 의미 있게 오른 변형은 없다.
- combined 규칙(세 모델 평균 +0.002 이상)을 넘은 블록이 없어서 합친 변형은 만들지 않았다. 가장 높은 +maccs도 평균 +0.0017이었다.

## 판단
1. 가설 1은 기각됐다. descriptor를 늘려도(RDKit 217, Mordred 700) 오르지 않고 오히려 평균 0.0024 낮아졌다.
   - 1차에서 연속형 변수의 gain이 컸던 건 clogp·TPSA·MolWt 같은 핵심 변수 몇 개 덕분이었다.
   - 이 변수들은 이미 base에 들어 있다. 새 descriptor는 대부분 이와 겹치거나 노이즈가 되어, 컬럼 샘플링(30%)에서 핵심 변수가 뽑힐 기회를 줄인 것으로 보인다.
2. 가설 2도 거의 효과가 없었다.
   - 입체 정보를 반영한 count Fingerprint는 RandomForest에서만 +0.0015였다.
   - 유사도 0.5 미만 구간의 정답률도 +maccs에서 0.737 → 0.744로 조금 오른 정도다.
3. 차원 축소는 트리에는 해롭다(LightGBM −0.015, 유의). 비트 하나하나로 나누는 트리의 장점이 사라진다. MLP에서만 약간 올랐지만 유의하지 않다.
4. 결론적으로 입력 표현은 포화 상태에 가깝다. 구조가 거의 같은 화합물(유사도 0.9 이상)끼리도 라벨 일치율이 0.81이었던 것과 같은 맥락이다.

## 앙상블 재료로서의 가치 (4차 근거)
- 다른 입력으로 학습한 모델은 base RandomForest와 예측 순위 상관이 0.91~0.94다. 같은 입력의 다른 트리(0.94~0.98)보다 서로 다르다.
- base RandomForest와 단순 평균한 결과: +ap_tt LightGBM 0.8219(+0.0042), +ptfp LightGBM 0.8202, +mordred XGBoost 0.8197.
- 1차에서 같은 입력끼리 섞었을 때(+0.0015)보다 효과가 크다. 4차에서 입력이 다른 모델들을 섞는 방향이 가장 유망하다.

## 3차로 넘길 것
- 튜닝할 입력: +maccs (트리 3개 평균 F1 최고, 1,167컬럼). 차이가 작아서 컬럼이 적고 계산이 빠른 것도 장점이다.
- 4차 앙상블 후보: +ap_tt·+ptfp·+mordred 모델처럼 입력이 다른 모델의 OOF 예측 (`results/oof_predictions.csv`)

## 제출 파일
- `predict_output_임채현_2차.csv`: 2차 단일 최고 (count_chiral × RandomForest, CV F1 0.8191)를 8,000행 전체로 재학습해 예측했다. 테스트 세트(1,600행) F1 0.8208, AUC 0.875 (1차 RandomForest 테스트 F1 0.813).

## 파일
| 파일 | 내용 |
|---|---|
| `featurize.py` | 추가 블록 계산과 정리 함수 (train·predict 공통) |
| `build_features.py`, `build_mordred_only.py` | 학습 데이터 블록 계산 → `cache/` |
| `exp02_feature_variants.ipynb` | 실험 노트북 (실행 결과 포함) |
| `models2.py` | 2차 모델 정의 (제출용) |
| `predict_submission.ipynb` | 제출 파일 생성 |
| `results/block_spec.json` | 블록별 남길 컬럼과 결측 대체값 |
| `results/variant_columns.json` | 변형별 컬럼 목록, 최고 변형 |
| `results/cv_scores.csv`, `oof_predictions.csv` | fold별 점수, OOF 예측 |
| `results/f1_by_variant.csv`, `paired_ttest_vs_base.csv` | 요약표, 짝지은 검정 |
| `results/fig_variant_f1.png` | 변형별 F1 그래프 |
