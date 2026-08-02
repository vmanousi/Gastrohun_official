import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Three-way comparison for the 8 DINO/DINOv2 models: frozen probe
# (unfrozen=0), the fair single-LR recipe (unfrozen=40), and the new
# discriminative-LR recipe (unfrozen=40, backbone_lr=1e-5, head_lr=7e-4,
# seed=42) -- all otherwise identical (dataset normalization, same batch
# sizes, same epochs/early-stopping). Answers: does discriminative LR
# actually help, and does it fix the two models that got worse under
# single-LR fine-tuning (dinov2_vits14, dino_vits16)?
# Only READS analysis/figure5_dino/data/ScenarioA-B_bootstrap.json (same
# 100 resamples used everywhere else) -- never writes there.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output"

VARIANT_ROOTS = {
    "frozen": OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen0",
    "single_lr": OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40",
    "discriminative_lr": OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr",
}

MODELS = [
    "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8",
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
]

PRED_COLS = ["set_type", "Complete agreement", "PredictedClass"]


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

print("Loading frozen / single_lr / discriminative_lr predictions for all 8 DINO/DINOv2 models...")
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
                "variant": variant,
                "iteration": row["iteration"],
                "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0) * 100,
            })

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df_combined = pd.DataFrame(records)
df_combined.to_csv(RESULTS_DIR / "discriminative_lr_bootstrap_metrics.csv", index=False)

print("\nPer-model comparison:")
rows = []
for model in MODELS:
    stats_by_variant = {}
    for variant in VARIANT_ROOTS:
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"]
        mean, margin = calculate_mean_and_margin_error(vals)
        stats_by_variant[variant] = (mean, margin)

    f_mean, f_margin = stats_by_variant["frozen"]
    s_mean, s_margin = stats_by_variant["single_lr"]
    d_mean, d_margin = stats_by_variant["discriminative_lr"]

    rows.append({
        "model": model,
        "frozen_mean": round(f_mean, 2),
        "single_lr_mean": round(s_mean, 2),
        "discriminative_lr_mean": round(d_mean, 2),
        "disclr_vs_singlelr_gain": round(d_mean - s_mean, 2),
        "disclr_vs_singlelr_ci_overlap": ci_overlap(d_mean, d_margin, s_mean, s_margin),
        "disclr_vs_frozen_gain": round(d_mean - f_mean, 2),
        "disclr_vs_frozen_ci_overlap": ci_overlap(d_mean, d_margin, f_mean, f_margin),
    })

df_summary = pd.DataFrame(rows).sort_values("disclr_vs_singlelr_gain", ascending=False)
df_summary.to_csv(RESULTS_DIR / "discriminative_lr_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nBuilding three-way comparison plot...")
model_order = df_summary["model"].tolist()
colors = {"frozen": "#1F77B4", "single_lr": "#7F7F7F", "discriminative_lr": "#D62728"}
labels = {"frozen": "Frozen (unfrozen=0)", "single_lr": "Single LR (unfrozen=40)", "discriminative_lr": "Discriminative LR"}

fig, ax = plt.subplots(figsize=(16, 7))
group_width = 0.85
variants = list(VARIANT_ROOTS.keys())
for i, model in enumerate(model_order):
    for j, variant in enumerate(variants):
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"].values
        pos = i + (j - 1) * (group_width / 3)
        color = colors[variant]
        ax.boxplot(vals, positions=[pos], widths=group_width / 3.5, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="none", edgecolor=color),
                   medianprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=4, color=color, alpha=0.5)

ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(model_order, rotation=30, ha="right")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("Frozen vs. single-LR vs. discriminative-LR fine-tuning, DINO/DINOv2\n"
              "(unfrozen=40 for both fine-tuned variants; backbone_lr=1e-5, head_lr=7e-4, seed=42)")
handles = [plt.Line2D([0], [0], color=c, lw=3, label=labels[v]) for v, c in colors.items()]
ax.legend(handles=handles, loc="lower left")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "discriminative_lr_comparison.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'discriminative_lr_comparison.png'}")
