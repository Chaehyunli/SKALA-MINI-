import sys, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import featurize as F
smi = pd.read_parquet(HERE / "cache" / "index.parquet")["SMILES"].tolist()
t = time.time(); F.mordred_desc(smi, nproc=2).to_parquet(HERE / "cache" / "mordred.parquet"); print("mordred", round(time.time()-t), "s", flush=True)
