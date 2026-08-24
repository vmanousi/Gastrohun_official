import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Scoped-down seed check for the two Optuna-confirmed results that came out
# WORSE than their previous best-known recipe (vit_b_16, dino_vits16): is
# that a real effect of the Optuna-found LR ratio, or noise from a single
# seed (42)? Compares seed=42 (already in Complete_agreement_40_optuna_
# confirmed) against a second independent run at seed=123, alongside each
# model's previous-best (old recipe) result for context. Not re-checking
# the DINOv2 wins -- their margins are large enough that a seed check
# there is low-value (see compare_optuna_confirmed.py's results).
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
OLD_REPO_ROOT = REPO_ROOT.parent / "Gastrohun_official"

BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output"
SEED42_ROOT = OUTPUT_ROOT / "Complete_agreement_40_optuna_confirmed"
SEED123_ROOT = OUTPUT_ROOT / "Complete_agreement_40_optuna_confirmed_seed123"

OLD_OUTPUT_ROOT = OLD_REPO_ROOT / "image_classification" / "output"
OLD_SOURCES = {
    "vit_b_16": OLD_OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40",
    "dino_vits16": OLD_OUTPUT_ROOT / "Complete_agreement_40_dino_midlr",
}

MODELS = ["vit_b_16", "dino_vits16"]
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

print("Loading previous-best, optuna_seed42, and optuna_seed123 predictions for vit_b_16 / dino_vits16...")
frames = []
for model in MODELS:
    frames.append(load_predictions(OLD_SOURCES[model], model, "previous_best"))
    frames.append(load_predictions(SEED42_ROOT, model, "optuna_seed42"))
    frames.append(load_predictions(SEED123_ROOT, model, "optuna_seed123"))
df_prediction = pd.concat(frames, ignore_index=True)

variants = ["previous_best", "optuna_seed42", "optuna_seed123"]

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
df_combined.to_csv(RESULTS_DIR / "vit_dino_seed_check_bootstrap_metrics.csv", index=False)

print("\nPer-model summary (previous-best recipe vs. Optuna recipe across 2 seeds):")
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

    seed_means = [stats_by_variant["optuna_seed42"][0], stats_by_variant["optuna_seed123"][0]]
    row["optuna_seed_mean"] = round(np.mean(seed_means), 2)
    row["optuna_seed_spread"] = round(max(seed_means) - min(seed_means), 2)

    old_mean, old_margin = stats_by_variant["previous_best"]
    lo_old, hi_old = old_mean - old_margin, old_mean + old_margin
    both_seeds_below_old = all(m < lo_old for m in seed_means)
    both_seeds_above_old = all(m > hi_old for m in seed_means)
    row["both_optuna_seeds_significantly_worse_than_previous_best"] = both_seeds_below_old
    row["both_optuna_seeds_significantly_better_than_previous_best"] = both_seeds_above_old
    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.to_csv(RESULTS_DIR / "vit_dino_seed_check_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nBuilding comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"previous_best": "#7F7F7F", "optuna_seed42": "#D62728", "optuna_seed123": "#E67E22"}
group_width = 0.85
for i, model in enumerate(MODELS):
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
        jitter = np.random.normal(0, 0.03, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=5, color=color, alpha=0.5)

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS, rotation=0, ha="center")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("Is the Optuna-recipe regression real? previous-best vs. 2 Optuna seeds")
handles = [plt.Line2D([0], [0], color=c, lw=3, label=v) for v, c in colors.items()]
ax.legend(handles=handles, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "vit_dino_seed_check.png", dpi=200)
print("Saved plot to {}".format(RESULTS_DIR / "vit_dino_seed_check.png"))
