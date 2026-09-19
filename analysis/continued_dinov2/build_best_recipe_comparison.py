"""EXPLORATORY / secondary analysis -- "what's each architecture's ceiling
after its own best-validated recipe", NOT the controlled comparison.

Fully isolated from build_continued_comparison.py (the PRIMARY, fixed-recipe
result the thesis's core claim rests on) and from
analysis/original16_repro/build_figure5_best_recipe.py (which this script
reads from, read-only, but never writes to). Own output files below.

Per-model tuning confounds "architecture quality" with "how much tuning
search was spent on this model" -- e.g. dino_vits16 alone gains +9.64 macro-F1
from recipe tuning, more than the whole continued-pretraining effect in the
primary comparison (+5.43). Report this table ONLY alongside that caveat,
never as a replacement for the fixed-recipe result.

Adds two new entries, using the EXACT SAME discriminative-LR recipe
(backbone_lr=1e-5, head_lr=7e-4, seed=42 fixed) that already fixed DINOv2's
"recipe-driven underperformance" for the other 4 DINOv2 variants in this
project (+7 to +9.4 macro F1 over single_lr):
  - dinov2_vits14_reg_generic    (same architecture as the primary comparison)
  - dinov2_vits14_reg_continued  (our continued-SSL checkpoint, best iter)
Both trained by cluster/slurm/train_continued_discriminative_lr.sh into its
own isolated output root (Complete_agreement_40_discriminative_lr_continued/).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parents[1]
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
OUTPUT_ROOT = REPO_ROOT / "image_classification" / "output"

SINGLE_LR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_repro_unfrozen40"
MID_LR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_dino_midlr"
DISCRIMINATIVE_LR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr"
CONTINUED_DISCLR_ROOT = OUTPUT_ROOT / "Complete_agreement_40_discriminative_lr_continued"  # new, isolated

BASELINE_MODELS = [
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vgg11", "vgg13", "vgg16",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]

# model -> (source root, recipe label, iter dir). Identical to
# build_figure5_best_recipe.py's BEST_RECIPE for the original 24 -- copied,
# not imported, so this file has zero runtime dependency on that one.
BEST_RECIPE = {}
for m in BASELINE_MODELS:
    BEST_RECIPE[m] = (SINGLE_LR_ROOT, "single_lr", "iter1")
BEST_RECIPE["dino_vits16"] = [(MID_LR_ROOT, "mid_lr", ["iter1"])]
for m in ["dino_vits8", "dino_vitb16", "dino_vitb8"]:
    BEST_RECIPE[m] = [(SINGLE_LR_ROOT, "single_lr", ["iter1"])]
for m in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]:
    BEST_RECIPE[m] = [(DISCRIMINATIVE_LR_ROOT, "discriminative_lr", ["iter1"])]
for m in BASELINE_MODELS:
    BEST_RECIPE[m] = [(SINGLE_LR_ROOT, "single_lr", ["iter1"])]

# our two new entries: DON'T assume discriminative_lr wins here just because it
# won for the other 4 DINOv2 variants -- pick dynamically between the two
# recipes we actually have results for (each candidate's own real bootstrap
# mean decides it, exactly like the rest of this table was decided).
for m in ["dinov2_vits14_reg_generic", "dinov2_vits14_reg_continued"]:
    BEST_RECIPE[m] = [
        (SINGLE_LR_ROOT, "single_lr", ["iter1", "iter2", "iter3"]),   # 3 seeds, primary comparison
        (CONTINUED_DISCLR_ROOT, "discriminative_lr", ["iter1"]),      # 1 seed, seed=42, this track
    ]

ALL_MODELS = list(BEST_RECIPE.keys())
NEW_MODELS = {"dinov2_vits14_reg_generic", "dinov2_vits14_reg_continued"}
PRED_COLS = ["set_type", "Complete agreement", "PredictedClass"]


def family_of(name):
    if name == "dinov2_vits14_reg_continued":
        return "DINOv2-continued"
    if name.startswith("dinov2"):
        return "DINOv2"
    if name.startswith("dino"):
        return "DINO"
    if name.startswith("convnext"):
        return "ConvNeXt"
    if name.startswith("resnet"):
        return "ResNet"
    if name.startswith("vgg"):
        return "VGG"
    if name.startswith("vit_"):
        return "VisionTransformer"
    return "Other"


def mean_and_margin(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sem = stats.sem(values)
    t_crit = stats.t.ppf(1 - alpha / 2, len(values) - 1)
    return mean, t_crit * sem


def load_predictions(root, model, iterdir):
    path = root / model / iterdir / "predict.json"
    if not path.exists():
        return None
    df = pd.read_json(path)
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
    return df.reset_index(drop=False)


def pooled_bootstrap_values(root, model, iterdirs, df_bootstrap):
    """Pooled per-resample macro-F1 across all seeds for one (model, recipe) candidate."""
    out = []
    for iterdir in iterdirs:
        df = load_predictions(root, model, iterdir)
        if df is None:
            return None
        by_index = df.set_index("index")
        for _, row in df_bootstrap.iterrows():
            sel = by_index.loc[by_index.index.intersection(row["test_index"])]
            out.append(f1_score(sel["Complete agreement"].astype(np.int64),
                                sel["PredictedClass"].astype(np.int64),
                                average="macro", zero_division=0) * 100)
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_bootstrap = pd.read_json(BOOTSTRAP_PATH)

    print("EXPLORATORY best-recipe-per-model comparison -- NOT the controlled result.")
    print("Resolving each model's best-validated recipe (highest bootstrap mean among tried candidates)...\n")

    records, resolved, missing_models = [], {}, []
    for model in ALL_MODELS:
        best = None  # (mean, recipe, values)
        for root, recipe, iterdirs in BEST_RECIPE[model]:
            vals = pooled_bootstrap_values(root, model, iterdirs, df_bootstrap)
            if vals is None:
                print(f"  candidate missing: {model} ({recipe}) under {root.name}")
                continue
            mean = np.mean(vals)
            tag = "" if len(BEST_RECIPE[model]) == 1 else f"  [candidate: {recipe} = {mean:.2f}]"
            print(f"  {model} ({recipe}): {len(vals)} bootstrap draws, mean {mean:.2f}{tag}")
            if best is None or mean > best[0]:
                best = (mean, recipe, vals)
        if best is None:
            missing_models.append(model)
            continue
        _, recipe, vals = best
        resolved[model] = recipe
        for v in vals:
            records.append({"model": model, "family": family_of(model), "recipe": recipe, "macro_f1": v})

    df_combined = pd.DataFrame(records)
    df_combined.to_csv(RESULTS_DIR / "best_recipe_with_continued_bootstrap_metrics.csv", index=False)

    rows = []
    for model in ALL_MODELS:
        if model in missing_models:
            continue
        vals = df_combined[df_combined["model"] == model]["macro_f1"]
        mean, margin = mean_and_margin(vals)
        rows.append({"model": model, "family": family_of(model), "recipe": resolved[model],
                     "macro_f1_mean": round(mean, 2), "macro_f1_margin": round(margin, 2)})
    df_summary = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    df_summary.to_csv(RESULTS_DIR / "best_recipe_with_continued_summary.csv", index=False)

    print("\n=== EXPLORATORY RANKING -- each model's own best-validated recipe ===")
    print(df_summary.to_string(index=False))
    if missing_models:
        print(f"\n(not yet run: {missing_models} -- submit "
              f"cluster/slurm/train_continued_discriminative_lr.sh + test_continued_discriminative_lr.sh)")

    if not (set(NEW_MODELS) & set(df_summary["model"])):
        return
    print("\n=== continued vs generic, best-recipe (discriminative_lr) ===")
    print(df_summary[df_summary["model"].isin(NEW_MODELS)].to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = df_summary["model"].tolist()
    fig, ax = plt.subplots(figsize=(20, 8))
    for pos, model in enumerate(order):
        vals = df_combined[df_combined["model"] == model]["macro_f1"].values
        color = "#D62728" if model == "dinov2_vits14_reg_continued" else \
                "#2CA02C" if model == "dinov2_vits14_reg_generic" else "#888888"
        ax.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=("#ffe0e0" if model in NEW_MODELS else "none"), edgecolor=color),
                   medianprops=dict(color=color))
        ax.scatter(np.full(len(vals), pos) + np.random.normal(0, 0.06, len(vals)), vals,
                   s=4, color=color, alpha=0.5)
    ax.set_xticks(range(len(order)))
    labels = ax.set_xticklabels(order, rotation=60, ha="right")
    for lbl, m in zip(labels, order):
        if m in NEW_MODELS:
            lbl.set_fontweight("bold")
    ax.set_ylabel("Macro F1-score (%)")
    ax.set_title("EXPLORATORY -- best-validated recipe per model (not the controlled comparison)\n"
                 "continued-DINOv2 in red, generic-DINOv2-reg in green, both on discriminative_lr")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "best_recipe_with_continued.png", dpi=200)
    print(f"\nsaved {RESULTS_DIR / 'best_recipe_with_continued.png'}")


if __name__ == "__main__":
    main()
