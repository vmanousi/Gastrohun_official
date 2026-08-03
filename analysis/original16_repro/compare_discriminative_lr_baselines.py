import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Control experiment: does discriminative LR (backbone_lr=1e-5, head_lr=0.0007,
# seed=42, unfrozen=40) help the 16 baseline models too, or is it specific to
# DINOv2? Same single_lr vs discriminative_lr comparison as
# compare_discriminative_lr.py, but for ConvNeXt/ResNet/VGG/ViT instead of
# DINO/DINOv2. Only READS analysis/figure5_dino/data/ScenarioA-B_bootstrap.json
# (same 100 resamples used everywhere else) -- never writes there.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output"

VARIANT_ROOTS = {
    "single_lr": OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40",
    "discriminative_lr": OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr",
}

MODELS = [
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vgg11", "vgg13", "vgg16",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]

PRED_COLS = ["set_type", "Complete agreement", "PredictedClass"]


def family_of(model_name):
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


def ci_overlap(mean_a, margin_a, mean_b, margin_b):
    lo_a, hi_a = mean_a - margin_a, mean_a + margin_a
    lo_b, hi_b = mean_b - margin_b, mean_b + margin_b
    return not (hi_a < lo_b or hi_b < lo_a)


print("Loading the paper's original 100 bootstrap resamples (read-only reuse)...")
df_bootstrap = pd.read_json(BOOTSTRAP_PATH)

print("Loading single_lr / discriminative_lr predictions for all 16 baseline models...")
frames = []
for model in MODELS:
    for tag, root in VARIANT_ROOTS.items():
        frames.append(load_predictions(root, model, tag))
df_prediction = pd.concat(frames, ignore_index=True)

print("Computing bootstrap macro-F1 per variant per model...")
records = []
for model in MODELS:
    for variant in VARIANT_ROOTS:
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
df_combined.to_csv(RESULTS_DIR / "discriminative_lr_baselines_bootstrap_metrics.csv", index=False)

print("\nPer-model comparison:")
rows = []
for model in MODELS:
    s_vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == "single_lr")]["macro_f1"]
    d_vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == "discriminative_lr")]["macro_f1"]
    s_mean, s_margin = calculate_mean_and_margin_error(s_vals)
    d_mean, d_margin = calculate_mean_and_margin_error(d_vals)
    rows.append({
        "model": model,
        "family": family_of(model),
        "single_lr_mean": round(s_mean, 2),
        "discriminative_lr_mean": round(d_mean, 2),
        "disclr_gain": round(d_mean - s_mean, 2),
        "ci_overlap": ci_overlap(d_mean, d_margin, s_mean, s_margin),
    })

df_summary = pd.DataFrame(rows).sort_values("disclr_gain", ascending=False)
df_summary.to_csv(RESULTS_DIR / "discriminative_lr_baselines_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nAggregate discriminative-LR gain by family (baselines):")
print(df_summary.groupby("family")["disclr_gain"].mean().sort_values(ascending=False).to_string())

print("\nBuilding comparison plot...")
model_order = df_summary["model"].tolist()
colors = {"single_lr": "#7F7F7F", "discriminative_lr": "#D62728"}
labels = {"single_lr": "Single LR (unfrozen=40)", "discriminative_lr": "Discriminative LR"}

fig, ax = plt.subplots(figsize=(18, 7))
group_width = 0.8
for i, model in enumerate(model_order):
    for j, variant in enumerate(["single_lr", "discriminative_lr"]):
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
ax.set_xticklabels(model_order, rotation=45, ha="right")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("Baselines: single-LR vs. discriminative-LR fine-tuning (control for the DINOv2 result)\n"
              "backbone_lr=1e-5, head_lr=0.0007, seed=42, unfrozen=40 for both")
handles = [plt.Line2D([0], [0], color=c, lw=3, label=labels[v]) for v, c in colors.items()]
ax.legend(handles=handles, loc="upper right")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "discriminative_lr_baselines_comparison.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'discriminative_lr_baselines_comparison.png'}")
