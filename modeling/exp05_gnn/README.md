# 5차 실험: 그래프 신경망 (Chemprop D-MPNN) — 보류

## 상태
코드만 준비되어 있고 실행은 보류했다. 사용자 결정에 따라 6~9차를 먼저 진행했다.

## 계획
| 항목 | 내용 |
|---|---|
| 목적 | 1~4차 모델은 모두 Fingerprint·descriptor 기반이라 비슷하게 틀린다. 분자 그래프를 직접 학습하는 모델로 앙상블 다양성을 높인다 |
| 모델 | Chemprop 2 D-MPNN (메시지 전달 3단계, 은닉 300) + 분자 단위 특성 15개 (Descriptor 3 + RDKit 12) |
| 학습 | 25 epoch 고정, 배치 64, dropout 0.1, 같은 10 fold |
| 확인한 것 | fold 0에서 5 epoch만으로 F1 0.781, AUC 0.823. epoch당 약 15초 (CPU 2코어) |
| 예상 시간 | 교차검증 약 60분 + 제출 파일 약 35분 |

## 실행 방법 (재개할 때)
1. `pip install chemprop` (PyTorch 포함)
2. `exp05_gnn.ipynb` 실행 → `results/final_method.json` 생성
3. `python build_predict_nb.py` 후 `predict_submission.ipynb` 실행

## 판단
아직 실행 전이라 결과가 없다.

## 파일
| 파일 | 내용 |
|---|---|
| `gnn.py` | Chemprop 학습·예측 함수 |
| `ensemble.py` | fold 안 앙상블 평가 함수 |
| `build_nb.py`, `exp05_gnn.ipynb` | 실험 노트북 (실행 전) |
| `build_predict_nb.py` | 제출 노트북 생성기 |
