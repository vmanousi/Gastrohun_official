import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score

# This script is fully separate from analysis/figure5_dino/ — it only READS
# that folder's already-downloaded bootstrap indices (never writes there),
# and writes its own outputs under analysis/figure5_dino_imagenet_norm/results/.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"

DATASET_NORM_ROOT = REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40"
IMAGENET_NORM_ROOT = REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40_imagenet_norm"

MODELS = [
    "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8",
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
]

PRED_COLS = ["num patient", "filename", "set_type", "Complete agreement", "PredictedClass"]


def calculate_mean_and_margin_error(values, alpha=0.05):
    mean = np.mean(values)
    sem = stats.sem(values)
    t_crit = stats.t.ppf((1 + (1 - alpha)) / 2, len(values) - 1)
    margin = t_crit * sem
    return mean, margin


def load_predictions(root, model, tag):
    df = pd.read_json(root / model / "iter1" / "predict.json")
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
    df = df.reset_index(drop=False)
    df = df[PRED_COLS + ["index"]]
    df["variant"] = f"{model}_{tag}"
    df["base_model"] = model
    df["normalization"] = tag
    return df


print("Loading the paper's original 100 bootstrap resamples (read-only reuse)...")
df_bootstrap = pd.read_json(BOOTSTRAP_PATH)

print("Loading dataset-norm and imagenet-norm predictions for all 8 models...")
frames = []
for model in MODELS:
    frames.append(load_predictions(DATASET_NORM_ROOT, model, "dataset"))
    frames.append(load_predictions(IMAGENET_NORM_ROOT, model, "imagenet"))
df_prediction = pd.concat(frames, ignore_index=True)
print(f"Loaded {df_prediction['variant'].nunique()} variants, {len(df_prediction)} rows")

print("Computing bootstrap macro-F1 per variant per iteration...")
records = []
for variant in df_prediction["variant"].unique():
    df_variant = df_prediction[df_prediction["variant"] == variant]
    base_model = df_variant["base_model"].iloc[0]
    normalization = df_variant["normalization"].iloc[0]
    for _, row in df_bootstrap.iterrows():
        roi_idx = row["test_index"]
        df_res = df_variant[df_variant["index"].isin(roi_idx)]
        true_labels = df_res["Complete agreement"].values.astype(np.int64)
        pred_labels = df_res["PredictedClass"].values.astype(np.int64)
        records.append({
            "variant": variant,
            "base_model": base_model,
            "normalization": normalization,
            "iteration": row["iteration"],
            "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
        })

df_combined = pd.DataFrame(records)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df_combined.to_csv(RESULTS_DIR / "normalization_comparison_bootstrap_metrics.csv", index=False)

print("\nBuilding summary with overlap check...")
summary_rows = []
for model in MODELS:
    row = {"base_model": model}
    stats_by_norm = {}
    for norm in ["dataset", "imagenet"]:
        vals = df_combined[(df_combined["base_model"] == model) & (df_combined["normalization"] == norm)]["macro_f1"]
        mean, margin = calculate_mean_and_margin_error(vals)
        stats_by_norm[norm] = (mean, margin)
        row[f"{norm}_mean"] = mean
        row[f"{norm}_margin"] = margin
    d_mean, d_margin = stats_by_norm["dataset"]
    i_mean, i_margin = stats_by_norm["imagenet"]
    d_lo, d_hi = d_mean - d_margin, d_mean + d_margin
    i_lo, i_hi = i_mean - i_margin, i_mean + i_margin
    overlap = not (d_hi < i_lo or i_hi < d_lo)
    row["delta"] = i_mean - d_mean
    row["cis_overlap"] = overlap
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows).sort_values("delta", ascending=False)
df_summary.to_csv(RESULTS_DIR / "normalization_comparison_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nBuilding comparison plot...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

order = df_summary["base_model"].tolist()
colors = {"dataset": "#1F77B4", "imagenet": "#D62728"}

fig, ax = plt.subplots(figsize=(16, 7))
group_width = 0.8
for i, model in enumerate(order):
    for j, norm in enumerate(["dataset", "imagenet"]):
        vals = df_combined[(df_combined["base_model"] == model) & (df_combined["normalization"] == norm)]["macro_f1"].values
        pos = i + (j - 0.5) * (group_width / 2)
        color = colors[norm]
        ax.boxplot(vals, positions=[pos], widths=group_width / 2.3, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="none", edgecolor=color),
                   medianprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=5, color=color, alpha=0.5)

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("DINO/DINOv2: dataset-specific vs. ImageNet normalization\n(bootstrap distributions, same 100 resamples as Figure 5)")
handles = [plt.Line2D([0], [0], color=c, lw=3, label=("Dataset-norm (original)" if n == "dataset" else "ImageNet-norm (fixed)"))
           for n, c in colors.items()]
ax.legend(handles=handles, loc="lower left")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "normalization_comparison.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'normalization_comparison.png'}")
print(f"Saved summary to {RESULTS_DIR / 'normalization_comparison_summary.csv'}")
