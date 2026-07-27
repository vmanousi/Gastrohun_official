import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Computes the *actual* published macro F1 per model directly from the
# published per-image predictions (analysis/figure5_dino/data/ScenarioA_Predictions.json),
# rather than trusting the paper's tables (which only show the top 5 of 16
# models). Diffs that against your reproduced runs' metrics.csv, to check
# whether --unfrozen_layers 60 (or whatever UNFROZEN below is set to)
# actually reproduces the published numbers.
#
# Usage: python compare_repro.py [unfrozen_value]
# Reads image_classification/output/Complete_agreement_40_repro_unfrozen<value>/<model>/iter1/metrics.csv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PUBLISHED_DATA = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA_Predictions.json"

MODELS = [
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vgg11", "vgg13", "vgg16",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]


def published_macro_f1():
    df = pd.read_json(PUBLISHED_DATA)
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())]
    scores = {}
    for model in MODELS:
        d = df[df["architecture"] == model]
        true_labels = d["Complete agreement"].values.astype(np.int64)
        pred_labels = d["PredictedClass"].values.astype(np.int64)
        scores[model] = f1_score(true_labels, pred_labels, average="macro", zero_division=0) * 100
    return scores


def repro_macro_f1(unfrozen):
    output_root = REPO_ROOT / "image_classification" / "output" / f"Complete_agreement_40_repro_unfrozen{unfrozen}"
    scores = {}
    for model in MODELS:
        metrics_path = output_root / model / "iter1" / "metrics.csv"
        if not metrics_path.exists():
            scores[model] = None
            continue
        df = pd.read_csv(metrics_path, index_col=0)
        row = df[df["Metric"] == "Macro F1-score"]
        scores[model] = float(row["Value"].iloc[0]) if len(row) else None
    return scores


def main(unfrozen):
    pub = published_macro_f1()
    repro = repro_macro_f1(unfrozen)

    rows = []
    for model in MODELS:
        p = pub[model]
        r = repro[model]
        rows.append({
            "model": model,
            "published_macro_f1": round(p, 2),
            f"repro_unfrozen{unfrozen}_macro_f1": None if r is None else round(r, 2),
            "delta": None if r is None else round(r - p, 2),
        })
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    missing = df_out[df_out[f"repro_unfrozen{unfrozen}_macro_f1"].isnull()]["model"].tolist()
    if missing:
        print(f"\nNo reproduced metrics.csv yet for: {missing}")

    out_path = SCRIPT_DIR / f"comparison_unfrozen{unfrozen}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    unfrozen = sys.argv[1] if len(sys.argv) > 1 else "60"
    main(unfrozen)
