import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

# Full-parity reproduction check: computes bootstrap macro F1 (same 100
# resamples as Figure 5/Table 6) for the published per-image predictions and
# for one or more reproduced --unfrozen_layers variants, so the comparison
# uses the exact same methodology the paper itself used to produce Table 6 --
# not a single-point estimate. Answers two things per model: (1) which
# unfrozen fraction lands closer to the published number, (2) whether that
# fraction's 95% CI actually overlaps the published one (i.e. genuinely
# reproduces it, not just "close by eye").
#
# Usage: python compare_repro.py [unfrozen_value ...]
# Reads image_classification/output/Complete_agreement_40_repro_unfrozen<value>/<model>/iter1/predict.json
# Defaults to checking 60 alone if no values are given.

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
PUBLISHED_DATA = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA_Predictions.json"
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"

MODELS = [
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vgg11", "vgg13", "vgg16",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]

PRED_COLS = ["set_type", "Complete agreement", "PredictedClass"]


def calculate_mean_and_margin_error(values, alpha=0.05):
    mean = np.mean(values)
    sem = stats.sem(values)
    t_crit = stats.t.ppf((1 + (1 - alpha)) / 2, len(values) - 1)
    margin = t_crit * sem
    return mean, margin


def load_published():
    df = pd.read_json(PUBLISHED_DATA)
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
    frames = []
    for model in MODELS:
        d = df[df["architecture"] == model].copy()
        d["variant"] = "published"
        d["base_model"] = model
        frames.append(d[PRED_COLS + ["index", "variant", "base_model"]])
    return pd.concat(frames, ignore_index=True)


def load_reproduction(unfrozen):
    output_root = REPO_ROOT / "image_classification" / "output" / f"Complete_agreement_40_repro_unfrozen{unfrozen}"
    frames = []
    missing = []
    for model in MODELS:
        path = output_root / model / "iter1" / "predict.json"
        if not path.exists():
            missing.append(model)
            continue
        df = pd.read_json(path)
        df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
        df = df.reset_index(drop=False)
        df["variant"] = f"unfrozen{unfrozen}"
        df["base_model"] = model
        frames.append(df[PRED_COLS + ["index", "variant", "base_model"]])
    if missing:
        print(f"  [unfrozen{unfrozen}] no predict.json yet for: {missing}")
    return pd.concat(frames, ignore_index=True) if frames else None


def bootstrap_macro_f1(df_prediction, df_bootstrap):
    records = []
    for variant in df_prediction["variant"].unique():
        df_variant = df_prediction[df_prediction["variant"] == variant]
        for base_model in df_variant["base_model"].unique():
            df_model = df_variant[df_variant["base_model"] == base_model]
            for _, row in df_bootstrap.iterrows():
                roi_idx = row["test_index"]
                df_res = df_model[df_model["index"].isin(roi_idx)]
                true_labels = df_res["Complete agreement"].values.astype(np.int64)
                pred_labels = df_res["PredictedClass"].values.astype(np.int64)
                records.append({
                    "variant": variant,
                    "base_model": base_model,
                    "iteration": row["iteration"],
                    "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
                })
    return pd.DataFrame(records)


def main(unfrozen_values):
    print("Loading published predictions and the paper's original bootstrap resamples...")
    df_bootstrap = pd.read_json(BOOTSTRAP_PATH)
    frames = [load_published()]

    print("Loading reproduced predictions...")
    for unfrozen in unfrozen_values:
        df = load_reproduction(unfrozen)
        if df is not None:
            frames.append(df)

    df_prediction = pd.concat(frames, ignore_index=True)
    print(f"Loaded variants: {sorted(df_prediction['variant'].unique())}")

    print("Computing bootstrap macro-F1 (100 resamples) per variant per model...")
    df_combined = bootstrap_macro_f1(df_prediction, df_bootstrap)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(RESULTS_DIR / "repro_bootstrap_metrics.csv", index=False)

    stats_by_variant_model = {}
    for variant in df_combined["variant"].unique():
        for model in MODELS:
            vals = df_combined[(df_combined["variant"] == variant) & (df_combined["base_model"] == model)]["macro_f1"]
            if len(vals) == 0:
                continue
            mean, margin = calculate_mean_and_margin_error(vals)
            stats_by_variant_model[(variant, model)] = (mean, margin)

    rows = []
    for model in MODELS:
        pub_mean, pub_margin = stats_by_variant_model.get(("published", model), (None, None))
        row = {"model": model, "published_mean": round(pub_mean, 2), "published_margin": round(pub_margin, 2)}
        best_variant, best_abs_delta = None, None
        for unfrozen in unfrozen_values:
            key = (f"unfrozen{unfrozen}", model)
            if key not in stats_by_variant_model:
                row[f"unfrozen{unfrozen}_mean"] = None
                row[f"unfrozen{unfrozen}_delta"] = None
                row[f"unfrozen{unfrozen}_ci_overlap"] = None
                continue
            r_mean, r_margin = stats_by_variant_model[key]
            delta = r_mean - pub_mean
            pub_lo, pub_hi = pub_mean - pub_margin, pub_mean + pub_margin
            r_lo, r_hi = r_mean - r_margin, r_mean + r_margin
            overlap = not (pub_hi < r_lo or r_hi < pub_lo)
            row[f"unfrozen{unfrozen}_mean"] = round(r_mean, 2)
            row[f"unfrozen{unfrozen}_delta"] = round(delta, 2)
            row[f"unfrozen{unfrozen}_ci_overlap"] = overlap
            if best_abs_delta is None or abs(delta) < best_abs_delta:
                best_abs_delta = abs(delta)
                best_variant = unfrozen
        row["closest_unfrozen"] = best_variant
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(RESULTS_DIR / "repro_summary.csv", index=False)
    print("\nPer-model comparison:")
    print(df_summary.to_string(index=False))

    print("\nAggregate verdict per unfrozen value:")
    for unfrozen in unfrozen_values:
        deltas = df_summary[f"unfrozen{unfrozen}_delta"].dropna()
        overlaps = df_summary[f"unfrozen{unfrozen}_ci_overlap"].dropna()
        if len(deltas) == 0:
            print(f"  unfrozen{unfrozen}: no reproduced data yet")
            continue
        print(f"  unfrozen{unfrozen}: mean|delta|={deltas.abs().mean():.2f}, "
              f"CI overlaps published on {int(overlaps.sum())}/{len(overlaps)} models")

    print(f"\nSaved {RESULTS_DIR / 'repro_summary.csv'} and {RESULTS_DIR / 'repro_bootstrap_metrics.csv'}")


if __name__ == "__main__":
    values = sys.argv[1:] if len(sys.argv) > 1 else ["60"]
    main(values)
