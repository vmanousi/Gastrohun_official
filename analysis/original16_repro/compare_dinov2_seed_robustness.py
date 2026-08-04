import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Robustness check for the headline discriminative-LR result on DINOv2: the
# original run used a single seed (42) and showed a +7 to +9.4 macro F1 gain
# over single-LR. Is that a real effect or a lucky seed? Reruns dinov2_vits14
# (smallest) and dinov2_vitg14 (largest) with two additional seeds (123, 7)
# under the identical discriminative-LR recipe (backbone=1e-5, head=7e-4,
# unfrozen=40) and compares spread across seeds against the original
# single_lr-vs-discriminative_lr gap.
# Only READS analysis/figure5_dino/data/ScenarioA-B_bootstrap.json (same
# 100 resamples used everywhere else) -- never writes there.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output"

MODELS = ["dinov2_vits14", "dinov2_vitg14"]
SEEDS = [42, 123, 7]

SINGLE_LR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40"


def seed_root(seed):
    if seed == 42:
        return OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr"
    return OUTPUT_ROOT / f"Complete_agreement_40_discriminative_lr_seed{seed}"


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


print("Loading the paper's original 100 bootstrap resamples (read-only reuse)...")
df_bootstrap = pd.read_json(BOOTSTRAP_PATH)

print("Loading single_lr baseline plus discriminative_lr across 3 seeds for dinov2_vits14 / dinov2_vitg14...")
frames = []
for model in MODELS:
    frames.append(load_predictions(SINGLE_LR_ROOT, model, "single_lr"))
    for seed in SEEDS:
        frames.append(load_predictions(seed_root(seed), model, f"discriminative_lr_seed{seed}"))
df_prediction = pd.concat(frames, ignore_index=True)

variants = ["single_lr"] + [f"discriminative_lr_seed{s}" for s in SEEDS]

print("Computing bootstrap macro-F1 per variant per model...")
records = []
for model in MODELS:
    for variant in variants:
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
df_combined.to_csv(RESULTS_DIR / "dinov2_seed_robustness_bootstrap_metrics.csv", index=False)

print("\nPer-model, per-seed summary:")
rows = []
for model in MODELS:
    stats_by_variant = {}
    for variant in variants:
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"]
        mean, margin = calculate_mean_and_margin_error(vals)
        stats_by_variant[variant] = (mean, margin)

    row = {"model": model}
    for variant in variants:
        mean, margin = stats_by_variant[variant]
        row[f"{variant}_mean"] = round(mean, 2)
        row[f"{variant}_margin"] = round(margin, 2)

    seed_means = [stats_by_variant[f"discriminative_lr_seed{s}"][0] for s in SEEDS]
    row["seed_mean"] = round(np.mean(seed_means), 2)
    row["seed_spread"] = round(max(seed_means) - min(seed_means), 2)

    single_mean, single_margin = stats_by_variant["single_lr"]
    lo_single, hi_single = single_mean - single_margin, single_mean + single_margin
    all_seeds_beat_single = all(m > hi_single for m in seed_means)
    row["all_seeds_exceed_single_lr_ci"] = all_seeds_beat_single
    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.to_csv(RESULTS_DIR / "dinov2_seed_robustness_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nBuilding seed-spread comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))
group_width = 0.85
for i, model in enumerate(MODELS):
    for j, variant in enumerate(variants):
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"].values
        pos = i + (j - 1.5) * (group_width / 4)
        color = "#7F7F7F" if variant == "single_lr" else "#D62728"
        ax.boxplot(vals, positions=[pos], widths=group_width / 4.5, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="none", edgecolor=color),
                   medianprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        jitter = np.random.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=4, color=color, alpha=0.5)

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS, rotation=0, ha="center")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("DINOv2 discriminative-LR: seed robustness (single_lr vs. 3 seeds of discriminative_lr)")
handles = [
    plt.Line2D([0], [0], color="#7F7F7F", lw=3, label="single_lr"),
    plt.Line2D([0], [0], color="#D62728", lw=3, label="discriminative_lr (seeds 42/123/7)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "dinov2_seed_robustness.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'dinov2_seed_robustness.png'}")
