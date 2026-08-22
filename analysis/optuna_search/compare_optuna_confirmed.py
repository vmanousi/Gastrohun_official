import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compares Optuna-confirmed (full-budget, equal-tuning-effort) results for
# 1 supervised ViT + 5 DINO/DINOv2 models against each model's previous
# best-known recipe. Answers the question this whole track exists to
# answer: does DINO/DINOv2 outperform ViT when BOTH get equal Optuna
# tuning effort, not just when DINO/DINOv2 is tuned and ViT is left on the
# paper's untouched default?
#
# The "previous best" predictions live in the OTHER, original clone
# (~/Gastrohun_official on the cluster) -- read cross-repo, READ-ONLY.
# This branch's isolation is about protecting that clone from writes, not
# reads: nothing here modifies anything in the sibling repo.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
OLD_REPO_ROOT = REPO_ROOT.parent / "Gastrohun_official"

BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
NEW_ROOT = REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40_optuna_confirmed"

OLD_OUTPUT_ROOT = OLD_REPO_ROOT / "image_classification" / "output"
OLD_SOURCES = {
    "vit_b_16": OLD_OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40",
    "dino_vits16": OLD_OUTPUT_ROOT / "Complete_agreement_40_dino_midlr",
    "dino_vitb8": OLD_OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40",
    "dinov2_vits14": OLD_OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr",
    "dinov2_vitl14": OLD_OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr",
    "dinov2_vitg14": OLD_OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr",
}

MODELS = ["vit_b_16", "dino_vits16", "dino_vitb8", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"]

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

print("Loading Optuna-confirmed and previous-best predictions for 6 models...")
frames = []
for model in MODELS:
    frames.append(load_predictions(NEW_ROOT, model, "optuna_confirmed"))
    frames.append(load_predictions(OLD_SOURCES[model], model, "previous_best"))
df_prediction = pd.concat(frames, ignore_index=True)

print("Computing bootstrap macro-F1 per variant per model...")
records = []
for model in MODELS:
    for variant in ["previous_best", "optuna_confirmed"]:
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
df_combined.to_csv(RESULTS_DIR / "optuna_confirmed_bootstrap_metrics.csv", index=False)

print("\nPrevious-best vs. Optuna-confirmed, per model:")
rows = []
for model in MODELS:
    stats_by_variant = {}
    for variant in ["previous_best", "optuna_confirmed"]:
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"]
        mean, margin = calculate_mean_and_margin_error(vals)
        stats_by_variant[variant] = (mean, margin)
    old_mean, old_margin = stats_by_variant["previous_best"]
    new_mean, new_margin = stats_by_variant["optuna_confirmed"]
    lo_old, hi_old = old_mean - old_margin, old_mean + old_margin
    lo_new, hi_new = new_mean - new_margin, new_mean + new_margin
    overlap = not (hi_old < lo_new or hi_new < lo_old)
    rows.append({
        "model": model,
        "previous_best_mean": round(old_mean, 2),
        "optuna_confirmed_mean": round(new_mean, 2),
        "gain": round(new_mean - old_mean, 2),
        "ci_overlap": overlap,
    })
df_summary = pd.DataFrame(rows)
df_summary.to_csv(RESULTS_DIR / "optuna_confirmed_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nHead-to-head: Optuna-tuned ViT vs. Optuna-tuned DINO/DINOv2 "
      "(the core question this track exists to answer):")
vit_vals = df_combined[(df_combined["model"] == "vit_b_16") & (df_combined["variant"] == "optuna_confirmed")]["macro_f1"]
vit_mean, vit_margin = calculate_mean_and_margin_error(vit_vals)
vit_lo, vit_hi = vit_mean - vit_margin, vit_mean + vit_margin
print("vit_b_16 (Optuna-tuned): {:.2f} +/- {:.2f}".format(vit_mean, vit_margin))

head_to_head = []
for model in MODELS:
    if model == "vit_b_16":
        continue
    vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == "optuna_confirmed")]["macro_f1"]
    mean, margin = calculate_mean_and_margin_error(vals)
    lo, hi = mean - margin, mean + margin
    overlap = not (hi < vit_lo or vit_hi < lo)
    beats_vit_significantly = mean > vit_hi and not overlap
    head_to_head.append({
        "model": model,
        "macro_f1": round(mean, 2),
        "margin": round(margin, 2),
        "beats_tuned_vit_significantly": beats_vit_significantly,
        "ci_overlap_with_vit": overlap,
    })
df_h2h = pd.DataFrame(head_to_head)
df_h2h.to_csv(RESULTS_DIR / "optuna_confirmed_vs_vit_head_to_head.csv", index=False)
print(df_h2h.to_string(index=False))

print("\nBuilding comparison plot...")
fig, ax = plt.subplots(figsize=(12, 7))
colors = {"previous_best": "#7F7F7F", "optuna_confirmed": "#D62728"}
group_width = 0.7
for i, model in enumerate(MODELS):
    for j, variant in enumerate(["previous_best", "optuna_confirmed"]):
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"].values
        pos = i + (j - 0.5) * (group_width / 2)
        color = colors[variant]
        ax.boxplot(vals, positions=[pos], widths=group_width / 2.2, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="none", edgecolor=color),
                   medianprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        jitter = np.random.normal(0, 0.03, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=5, color=color, alpha=0.5)

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS, rotation=30, ha="right")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("Optuna-confirmed (equal tuning effort) vs. previous best-known recipe")
handles = [
    plt.Line2D([0], [0], color="#7F7F7F", lw=3, label="Previous best-known recipe"),
    plt.Line2D([0], [0], color="#D62728", lw=3, label="Optuna-confirmed (this track)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "optuna_confirmed_comparison.png", dpi=200)
print("Saved plot to {}".format(RESULTS_DIR / "optuna_confirmed_comparison.png"))
