import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# DINO v1 only: all four points now tested on the "how fast should the
# backbone move" axis -- frozen (0), mid_lr (1e-4, 10x slower than head),
# single_lr (7e-4, same speed as head), discriminative_lr (1e-5, ~70x
# slower). Answers: does DINO v1 have its own optimum between the two
# previously-tested extremes, or is single_lr (full speed) genuinely best?
# Only READS analysis/figure5_dino/data/ScenarioA-B_bootstrap.json (same
# 100 resamples used everywhere else) -- never writes there.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output"

VARIANT_ROOTS = {
    "frozen": OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen0",
    "mid_lr": OUTPUT_ROOT / "Complete_agreement_40_dino_midlr",
    "single_lr": OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40",
    "discriminative_lr": OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr",
}
VARIANT_LABELS = {
    "frozen": "Frozen (backbone LR = 0)",
    "mid_lr": "Mid-LR (backbone = 1e-4, 10x slower)",
    "single_lr": "Single-LR (backbone = head = 7e-4)",
    "discriminative_lr": "Discriminative-LR (backbone = 1e-5, ~70x slower)",
}

MODELS = ["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"]

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

print("Loading frozen / mid_lr / single_lr / discriminative_lr predictions for the 4 DINO v1 models...")
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
df_combined.to_csv(RESULTS_DIR / "dino_lr_sweep_bootstrap_metrics.csv", index=False)

print("\nPer-model comparison (all four points on the backbone-speed axis):")
rows = []
for model in MODELS:
    stats_by_variant = {}
    for variant in VARIANT_ROOTS:
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"]
        mean, margin = calculate_mean_and_margin_error(vals)
        stats_by_variant[variant] = (mean, margin)

    row = {"model": model}
    for variant in VARIANT_ROOTS:
        mean, margin = stats_by_variant[variant]
        row[f"{variant}_mean"] = round(mean, 2)

    s_mean, s_margin = stats_by_variant["single_lr"]
    m_mean, m_margin = stats_by_variant["mid_lr"]
    lo_a, hi_a = m_mean - m_margin, m_mean + m_margin
    lo_b, hi_b = s_mean - s_margin, s_mean + s_margin
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    row["midlr_vs_singlelr_gain"] = round(m_mean - s_mean, 2)
    row["midlr_vs_singlelr_ci_overlap"] = overlap
    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.to_csv(RESULTS_DIR / "dino_lr_sweep_summary.csv", index=False)
print(df_summary.to_string(index=False))

print("\nBuilding four-point comparison plot...")
model_order = MODELS
variants = ["frozen", "mid_lr", "single_lr", "discriminative_lr"]
colors = {"frozen": "#1F77B4", "mid_lr": "#2CA02C", "single_lr": "#7F7F7F", "discriminative_lr": "#D62728"}

fig, ax = plt.subplots(figsize=(14, 7))
group_width = 0.85
for i, model in enumerate(model_order):
    for j, variant in enumerate(variants):
        vals = df_combined[(df_combined["model"] == model) & (df_combined["variant"] == variant)]["macro_f1"].values
        pos = i + (j - 1.5) * (group_width / 4)
        color = colors[variant]
        ax.boxplot(vals, positions=[pos], widths=group_width / 4.5, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="none", edgecolor=color),
                   medianprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        jitter = np.random.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=4, color=color, alpha=0.5)

ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(model_order, rotation=0, ha="center")
ax.set_ylabel("Macro F1-score (%)")
ax.set_title("DINO v1: full backbone-speed sweep (frozen -> mid-LR -> single-LR -> discriminative-LR)")
handles = [plt.Line2D([0], [0], color=c, lw=3, label=VARIANT_LABELS[v]) for v, c in colors.items()]
ax.legend(handles=handles, loc="lower left", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "dino_lr_sweep.png", dpi=200)
print(f"Saved plot to {RESULTS_DIR / 'dino_lr_sweep.png'}")
