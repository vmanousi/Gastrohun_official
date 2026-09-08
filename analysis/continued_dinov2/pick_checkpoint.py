"""Phase E step 4 -- read the val macro-F1 of the top-3 candidate continued-SSL
checkpoints (trained 1 seed each by train_continued_select.sh, both scenarios)
and print the winner to feed into train_continued_reg.sh (WINNER_ITER=...).

Selection is on VALIDATION only (history.xlsx / finetuning_val sheet); the test
split is never touched here.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "image_classification" / "output"
CANDS = sys.argv[1:]  # iter numbers, e.g. 39999 59999 79999
if not CANDS:
    sys.exit("usage: python pick_checkpoint.py 39999 59999 79999")


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
        h = OUT / f"Complete_agreement_40_repro_unfrozen{unf}" / f"dinov2_vits14_reg_cand{cand}" / "iter1" / "history.xlsx"
        v = best_val_f1(h) if h.exists() else float("nan")
        entry[f"val_f1_unfrozen{unf}"] = round(v, 2)
        total += v
    entry["val_f1_sum"] = round(total, 2)
    rows.append(entry)

df = pd.DataFrame(rows).sort_values("val_f1_sum", ascending=False)
print(df.to_string(index=False))
winner = df.iloc[0]["cand"]
print(f"\nWINNER_ITER={winner}")
print(f"-> WINNER_ITER={winner} sbatch cluster/slurm/train_continued_reg.sh")
