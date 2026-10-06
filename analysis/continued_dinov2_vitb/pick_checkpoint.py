"""Project 3 (ViT-B/14+reg4) stage 2 -- read the val macro-F1 of the top-3
candidate continued-SSL checkpoints (cluster/slurm/train_continued_vitb_select.sh,
both scenarios) and print the winner.

Selection is on VALIDATION only (history.xlsx / finetuning_val sheet); the test
split is never touched here.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "image_classification" / "output"
CANDS = sys.argv[1:]
if not CANDS:
    sys.exit("usage: python pick_checkpoint.py 89999 99999 49999")


def best_val_f1(history_xlsx):
    xl = pd.ExcelFile(history_xlsx)
    frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names
              if s.endswith("_val") and "val_f1_macro" in pd.read_excel(xl, sheet_name=s).columns]
    return max(df["val_f1_macro"].max() for df in frames) * 100


rows = []
for cand in CANDS:
    entry = {"cand": cand}
    total = 0.0
    for unf in (0, 40):
        h = OUT / f"Complete_agreement_40_repro_vitb_unfrozen{unf}" / f"dinov2_vitb14_reg_cand{cand}" / "iter1" / "history.xlsx"
        v = best_val_f1(h) if h.exists() else float("nan")
        entry[f"val_f1_unfrozen{unf}"] = round(v, 2)
        total += v
    entry["val_f1_sum"] = round(total, 2)
    rows.append(entry)

df = pd.DataFrame(rows).sort_values("val_f1_sum", ascending=False)
print(df.to_string(index=False))
winner = df.iloc[0]["cand"]
print(f"\nWINNER_ITER={winner}")
