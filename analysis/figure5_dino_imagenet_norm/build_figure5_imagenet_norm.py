import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Parallel to analysis/figure5_dino/build_updated_figure5.py, but pulls the
# DINO/DINOv2 predictions from Complete_agreement_40_imagenet_norm/ instead
# of Complete_agreement_40/. Only READS analysis/figure5_dino/data/ (the
# published Scenario A predictions + bootstrap indices) — never writes there,
# so the original updated_figure5.png / summary / metrics are untouched.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
ORIGINAL_DATA_DIR = REPO_ROOT / "analysis" / "figure5_dino" / "data"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40_imagenet_norm"

NEW_MODELS = [
    "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8",
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
]

PRED_COLS = ["num patient", "filename", "set_type", "Complete agreement",
             "PredictedClass", "MaxProbability", "ClassProbabilities"]


def family_of(model_name):
    if model_name.startswith("dinov2"):
        return "DINOv2 (ImageNet-norm)"
    if model_name.startswith("dino"):
        return "DINO (ImageNet-norm)"
    if model_name.startswith("convnext"):
        return "ConvNeXt"
    if model_name.startswith("resnet"):
        return "ResNet"
    if model_name.startswith("vgg"):
        return "VGG"
    if model_name.startswith("vit_"):
        return "VisionTransformer"
    return "Other"


def calculate_mean_and_margin_error(metric_roi, alpha=0.05):
    mean = np.mean(metric_roi)
    sem = stats.sem(metric_roi)
    t_crit = stats.t.ppf((1 + (1 - alpha)) / 2, len(metric_roi) - 1)
    margin = t_crit * sem
    return mean, margin


print("Loading published Scenario A predictions and bootstrap indices...")
df_prediction = pd.read_json(ORIGINAL_DATA_DIR / "ScenarioA_Predictions.json")
df_bootstrap = pd.read_json(ORIGINAL_DATA_DIR / "ScenarioA-B_bootstrap.json")
print(f"Original: {df_prediction['architecture'].nunique()} models, {len(df_prediction)} rows")

print("Loading and appending ImageNet-normalized DINO/DINOv2 predictions...")
new_frames = []
for model in NEW_MODELS:
    path = OUTPUT_ROOT / model / "iter1" / "predict.json"
    df = pd.read_json(path)
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
    df = df.reset_index(drop=False)
    df = df[PRED_COLS + ["index"]]
    df["architecture"] = model
    new_frames.append(df)
    print(f"  {model}: {len(df)} test rows")

df_prediction = pd.concat([df_prediction] + new_frames, ignore_index=True)
all_models = df_prediction["architecture"].unique()
print(f"Combined: {len(all_models)} models, {len(df_prediction)} rows")

print("Computing bootstrap macro-F1 per model per iteration...")
records = []
for model in all_models:
    df_model = df_prediction[df_prediction["architecture"] == model]
    for _, row in df_bootstrap.iterrows():
        roi_idx = row["test_index"]
        df_res = df_model[df_model["index"].isin(roi_idx)]
        true_labels = df_res["Complete agreement"].values.astype(np.int64)
        pred_labels = df_res["PredictedClass"].values.astype(np.int64)
        records.append({
            "model": model,
            "family": family_of(model),
            "iteration": row["iteration"],
            "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
            "macro_precision": precision_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
            "macro_recall": recall_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
        })

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df_combined = pd.DataFrame(records)
df_combined.to_csv(RESULTS_DIR / "figure5_imagenet_norm_bootstrap_metrics.csv", index=False)

mean_order = df_combined.groupby("model")["macro_f1"].mean().sort_values(ascending=False).index.tolist()

print("\nTop 10 by mean bootstrap macro F1:")
summary_rows = []
for model in mean_order:
    vals = df_combined[df_combined["model"] == model]["macro_f1"]
    mean, margin = calculate_mean_and_margin_error(vals)
    summary_rows.append({"model": model, "family": family_of(model), "macro_f1_mean": mean, "macro_f1_margin": margin})
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(RESULTS_DIR / "figure5_imagenet_norm_summary.csv", index=False)
print(df_summary.head(10).to_string(index=False))
print("\nFull ranking:")
print(df_summary.to_string(index=False))

print("\nBuilding Figure 5 (ImageNet-norm variant)...")
family_colors = {
    "ConvNeXt": "#8B1E3F",
    "ResNet": "#1F77B4",
    "VGG": "#6A3D9A",
    "VisionTransformer": "#2CA02C",
    "DINO (ImageNet-norm)": "#E67E22",
    "DINOv2 (ImageNet-norm)": "#D62728",
}

fig, ax = plt.subplots(figsize=(20, 8))
positions = range(len(mean_order))
for pos, model in zip(positions, mean_order):
    vals = df_combined[df_combined["model"] == model]["macro_f1"].values
    color = family_colors.get(family_of(model), "#777777")
    ax.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor="none", edgecolor=color),
               medianprops=dict(color=color),
               whiskerprops=dict(color=color),
               capprops=dict(color=color))
    jitter = np.random.normal(0, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=6, color=color, alpha=0.6)

ax.set_xticks(list(positions))
ax.set_xticklabels(mean_order, rotation=60, ha="right")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("Figure 5 (ImageNet-normalized DINO/DINOv2): Bootstrap distribution of Macro F1-score rankings\n(original 16 models + DINO/DINOv2 with corrected normalization)")

handles = [plt.Line2D([0], [0], color=c, lw=3, label=fam) for fam, c in family_colors.items()
           if fam in {family_of(m) for m in mean_order}]
ax.legend(handles=handles, loc="lower left")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "figure5_imagenet_norm.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'figure5_imagenet_norm.png'}")
