import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consolidation: the best VALIDATED recipe per model, all on one bootstrap-
# comparable ranking. Not a new experiment -- every prediction file here
# already exists from a completed, analyzed run:
#   - 16 baselines + dino_vits8/vitb16/vitb8: single_lr (unfrozen=40),
#     from build_figure5_repro.py's source -- full speed is their best.
#   - dino_vits16: mid_lr (backbone=1e-4, unfrozen=40), from
#     compare_dino_lr_sweep.py -- the only DINO v1 variant that benefits
#     from a slower backbone.
#   - all 4 DINOv2 variants: discriminative_lr (backbone=1e-5, unfrozen=40),
#     from compare_discriminative_lr.py -- the recipe that fixed DINOv2's
#     recipe-driven underperformance (+7 to +9.4 macro F1 over single_lr).
# Only READS analysis/figure5_dino/data/ScenarioA-B_bootstrap.json (same
# 100 resamples used everywhere else in this project) -- never writes there.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output"

SINGLE_LR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40"
MID_LR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_dino_midlr"
DISCRIMINATIVE_LR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr"

BASELINE_MODELS = [
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vgg11", "vgg13", "vgg16",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]

# model -> (source root, recipe label)
BEST_RECIPE = {}
for m in BASELINE_MODELS:
    BEST_RECIPE[m] = (SINGLE_LR_ROOT, "single_lr")
BEST_RECIPE["dino_vits16"] = (MID_LR_ROOT, "mid_lr")
for m in ["dino_vits8", "dino_vitb16", "dino_vitb8"]:
    BEST_RECIPE[m] = (SINGLE_LR_ROOT, "single_lr")
for m in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]:
    BEST_RECIPE[m] = (DISCRIMINATIVE_LR_ROOT, "discriminative_lr")

ALL_MODELS = list(BEST_RECIPE.keys())

PRED_COLS = ["set_type", "Complete agreement", "PredictedClass"]


def family_of(model_name):
    if model_name.startswith("dinov2"):
        return "DINOv2"
    if model_name.startswith("dino"):
        return "DINO"
    if model_name.startswith("convnext"):
        return "ConvNeXt"
    if model_name.startswith("resnet"):
        return "ResNet"
    if model_name.startswith("vgg"):
        return "VGG"
    if model_name.startswith("vit_"):
        return "VisionTransformer"
    return "Other"


def calculate_mean_and_margin_error(values, alpha=0.05):
    mean = np.mean(values)
    sem = stats.sem(values)
    t_crit = stats.t.ppf((1 + (1 - alpha)) / 2, len(values) - 1)
    margin = t_crit * sem
    return mean, margin


print("Loading the paper's original 100 bootstrap resamples (read-only reuse)...")
df_bootstrap = pd.read_json(BOOTSTRAP_PATH)

print("Loading predictions using each model's best-validated recipe...")
frames = []
for model in ALL_MODELS:
    root, recipe = BEST_RECIPE[model]
    path = root / model / "iter1" / "predict.json"
    df = pd.read_json(path)
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
    df = df.reset_index(drop=False)
    df["architecture"] = model
    df["recipe"] = recipe
    frames.append(df[PRED_COLS + ["index", "architecture", "recipe"]])
    print(f"  {model} ({recipe}): {len(df)} test rows")

df_prediction = pd.concat(frames, ignore_index=True)
print(f"Combined: {df_prediction['architecture'].nunique()} models, {len(df_prediction)} rows")

print("Computing bootstrap macro-F1 per model per iteration...")
records = []
for model in ALL_MODELS:
    root, recipe = BEST_RECIPE[model]
    df_model = df_prediction[df_prediction["architecture"] == model]
    for _, row in df_bootstrap.iterrows():
        roi_idx = row["test_index"]
        df_res = df_model[df_model["index"].isin(roi_idx)]
        true_labels = df_res["Complete agreement"].values.astype(np.int64)
        pred_labels = df_res["PredictedClass"].values.astype(np.int64)
        records.append({
            "model": model,
            "family": family_of(model),
            "recipe": recipe,
            "iteration": row["iteration"],
            "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
            "macro_precision": precision_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
            "macro_recall": recall_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
        })

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df_combined = pd.DataFrame(records)
df_combined.to_csv(RESULTS_DIR / "figure5_best_recipe_bootstrap_metrics.csv", index=False)

mean_order = df_combined.groupby("model")["macro_f1"].mean().sort_values(ascending=False).index.tolist()

print("\nFull ranking (all 24, best validated recipe per model):")
summary_rows = []
for model in mean_order:
    vals = df_combined[df_combined["model"] == model]["macro_f1"]
    mean, margin = calculate_mean_and_margin_error(vals)
    summary_rows.append({
        "model": model,
        "family": family_of(model),
        "recipe": BEST_RECIPE[model][1],
        "macro_f1_mean": mean,
        "macro_f1_margin": margin,
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(RESULTS_DIR / "figure5_best_recipe_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nBuilding the best-recipe Figure 5...")
family_colors = {
    "ConvNeXt": "#8B1E3F",
    "ResNet": "#1F77B4",
    "VGG": "#6A3D9A",
    "VisionTransformer": "#2CA02C",
    "DINO": "#E67E22",
    "DINOv2": "#D62728",
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
ax.set_title("Figure 5 (best validated recipe per model): Bootstrap distribution of Macro F1-score rankings\n"
              "(baselines + DINO v1 (single/mid-LR) + DINOv2 (discriminative-LR), all unfrozen=40%)")

handles = [plt.Line2D([0], [0], color=c, lw=3, label=fam) for fam, c in family_colors.items()
           if fam in {family_of(m) for m in mean_order}]
ax.legend(handles=handles, loc="lower left")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "figure5_best_recipe.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'figure5_best_recipe.png'}")
