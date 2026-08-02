import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Frozen linear probe (unfrozen=0) vs the fair fine-tuned recipe (unfrozen=40),
# for all 24 models -- same script, same environment, only the freeze
# percentage differs. Diagnostic: does fine-tuning actually help each model,
# and by how much (with a real significance check, not just point deltas)?
# Only READS analysis/figure5_dino/data/ScenarioA-B_bootstrap.json (same 100
# resamples used everywhere else) -- never writes there.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
FROZEN_ROOT = REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40_repro_unfrozen0"
FINETUNED_ROOT = REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40_repro_unfrozen40"

BASELINE_MODELS = [
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vgg11", "vgg13", "vgg16",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]
DINO_MODELS = [
    "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8",
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
]
ALL_MODELS = BASELINE_MODELS + DINO_MODELS

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


def load_predictions(root, model, tag):
    path = root / model / "iter1" / "predict.json"
    df = pd.read_json(path)
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
    df = df.reset_index(drop=False)
    df["variant"] = tag
    df["base_model"] = model
    return df[PRED_COLS + ["index", "variant", "base_model"]]


print("Loading the paper's original 100 bootstrap resamples (read-only reuse)...")
df_bootstrap = pd.read_json(BOOTSTRAP_PATH)

print("Loading frozen (unfrozen=0) and fine-tuned (unfrozen=40) predictions for all 24 models...")
frames = []
for model in ALL_MODELS:
    frames.append(load_predictions(FROZEN_ROOT, model, "frozen"))
    frames.append(load_predictions(FINETUNED_ROOT, model, "finetuned"))
df_prediction = pd.concat(frames, ignore_index=True)

print("Computing bootstrap macro-F1 per variant per model...")
records = []
for model in ALL_MODELS:
    for variant in ["frozen", "finetuned"]:
        df_model = df_prediction[(df_prediction["base_model"] == model) & (df_prediction["variant"] == variant)]
        for _, row in df_bootstrap.iterrows():
            roi_idx = row["test_index"]
            df_res = df_model[df_model["index"].isin(roi_idx)]
            true_labels = df_res["Complete agreement"].values.astype(np.int64)
            pred_labels = df_res["PredictedClass"].values.astype(np.int64)
            records.append({
                "model": model,
                "family": family_of(model),
                "variant": variant,
                "iteration": row["iteration"],
                "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
            })

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df_combined = pd.DataFrame(records)
df_combined.to_csv(RESULTS_DIR / "frozen_vs_finetuned_bootstrap_metrics.csv", index=False)

print("\nPer-model comparison:")
rows = []
for model in ALL_MODELS:
    f_vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == "frozen")]["macro_f1"]
    ft_vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == "finetuned")]["macro_f1"]
    f_mean, f_margin = calculate_mean_and_margin_error(f_vals)
    ft_mean, ft_margin = calculate_mean_and_margin_error(ft_vals)
    f_lo, f_hi = f_mean - f_margin, f_mean + f_margin
    ft_lo, ft_hi = ft_mean - ft_margin, ft_mean + ft_margin
    overlap = not (f_hi < ft_lo or ft_hi < f_lo)
    rows.append({
        "model": model,
        "family": family_of(model),
        "frozen_mean": round(f_mean, 2),
        "frozen_margin": round(f_margin, 2),
        "finetuned_mean": round(ft_mean, 2),
        "finetuned_margin": round(ft_margin, 2),
        "finetune_gain": round(ft_mean - f_mean, 2),
        "ci_overlap": overlap,
    })

df_summary = pd.DataFrame(rows).sort_values("finetune_gain", ascending=True)
df_summary.to_csv(RESULTS_DIR / "frozen_vs_finetuned_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nAggregate finetune_gain by family (mean, smallest gain = fine-tuning helps least / frozen features already strong):")
print(df_summary.groupby("family")["finetune_gain"].mean().sort_values().to_string())

print("\nBuilding comparison plot...")
model_order = df_summary.sort_values("finetuned_mean", ascending=False)["model"].tolist()
colors = {"frozen": "#1F77B4", "finetuned": "#D62728"}

fig, ax = plt.subplots(figsize=(20, 8))
group_width = 0.8
for i, model in enumerate(model_order):
    for j, variant in enumerate(["frozen", "finetuned"]):
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"].values
        pos = i + (j - 0.5) * (group_width / 2)
        color = colors[variant]
        ax.boxplot(vals, positions=[pos], widths=group_width / 2.3, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="none", edgecolor=color),
                   medianprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=4, color=color, alpha=0.5)

ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(model_order, rotation=60, ha="right")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("Frozen linear probe (unfrozen=0) vs. fair fine-tuned recipe (unfrozen=40)\n"
              "all 24 models, same environment/recipe otherwise")
handles = [plt.Line2D([0], [0], color=c, lw=3, label=("Frozen probe" if v == "frozen" else "Fine-tuned (40%)"))
           for v, c in colors.items()]
ax.legend(handles=handles, loc="upper right")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "frozen_vs_finetuned.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'frozen_vs_finetuned.png'}")
