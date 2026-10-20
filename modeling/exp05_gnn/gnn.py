"""
5차: Chemprop D-MPNN (방향성 메시지 전달 그래프 신경망) 학습 · 예측 함수
- 입력: SMILES로 만든 분자 그래프 (원자 = 노드, 결합 = 엣지) + 선택적으로 분자 단위 특성(x_d, 연속형 15개)
- x_d 표준화는 학습 fold로만 fit
"""
import logging, warnings
import numpy as np
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
torch.set_num_threads(2)
FEAT = featurizers.SimpleMoleculeMolGraphFeaturizer()


def _dataset(smiles, y=None, xd=None):
    dps = [data.MoleculeDatapoint.from_smi(s, None if y is None else [float(y[i])],
                                           x_d=None if xd is None else xd[i].astype(np.float32))
           for i, s in enumerate(smiles)]
    return data.MoleculeDataset(dps, FEAT)


def fit_predict(smi_tr, y_tr, smi_te, xd_tr=None, xd_te=None, epochs=30, d_h=300, depth=3,
                dropout=0.1, batch_size=64, seed=0):
    """학습 후 smi_te의 비독성(1) 확률을 반환"""
    pl.seed_everything(seed, workers=True, verbose=False)
    if xd_tr is not None:
        sc = StandardScaler().fit(xd_tr)
        xd_tr, xd_te = sc.transform(xd_tr), sc.transform(xd_te)
    tr = _dataset(smi_tr, y_tr, xd_tr); te = _dataset(smi_te, None, xd_te)
    tl = data.build_dataloader(tr, batch_size=batch_size, num_workers=0, shuffle=True, seed=seed)
    vl = data.build_dataloader(te, batch_size=256, num_workers=0, shuffle=False)
    mp = nn.BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout)
    n_xd = 0 if xd_tr is None else xd_tr.shape[1]
    ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim + n_xd, hidden_dim=300, dropout=dropout)
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=True)
    tr_ = pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False, deterministic=False)
    tr_.fit(model, tl)
    pred = tr_.predict(model, vl)
    return torch.cat(pred).numpy().ravel()
