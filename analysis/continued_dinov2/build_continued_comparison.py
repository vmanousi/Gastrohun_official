"""Phase E -- slot continued-DINOv2 onto the fair-comparison axis.

Reuses the paper's 100 bootstrap resamples (analysis/figure5_dino/data/
ScenarioA-B_bootstrap.json) and the same recipe/output layout as
compare_frozen_vs_finetuned.py. Reads:

  - the 24 baseline models   (iter1, as everywhere else)
  - dinov2_vits14_reg_generic    (iter1/2/3 -- 3 seeds)
  - dinov2_vits14_reg_continued  (iter1/2/3 -- 3 seeds)

both under Complete_agreement_40_repro_unfrozen{0,40}. For the 3-seed entries
each seed contributes its own 100 bootstrap macro-F1 values; the entry's
distribution is the pooled 300 (bootstrap + seed variance).

Outputs (results/):
  table_A_all_models.csv       every model, frozen + finetuned, mean +/- 95% margin
  table_B_ssl_only.csv         DINO v1 + DINOv2 + generic_reg + continued only
  continued_frozen.png         frozen axis, all models ranked, continued highlighted
  continued_finetuned.png      finetuned axis, same
  continued_bootstrap_metrics.csv   raw per-resample values
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parents[1]
BOOTSTRAP_PATH = REPO_ROOT / "analysis" / "figure5_dino" / "data" / "ScenarioA-B_bootstrap.json"
ROOTS = {
    "frozen": REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40_repro_unfrozen0",
    "finetuned": REPO_ROOT / "image_classification" / "output" / "Complete_agreement_40_repro_unfrozen40",
}

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
# (name, list-of-iter-dirs). Baselines are single-seed iter1; the new reg
# entries are 3-seed.
SINGLE_SEED = [(m, ["iter1"]) for m in BASELINE_MODELS + DINO_MODELS]
MULTI_SEED = [
    ("dinov2_vits14_reg_generic", ["iter1", "iter2", "iter3"]),
    ("dinov2_vits14_reg_continued", ["iter1", "iter2", "iter3"]),
]
ALL_ENTRIES = SINGLE_SEED + MULTI_SEED
NEW_MODELS = {"dinov2_vits14_reg_generic", "dinov2_vits14_reg_continued"}
SSL_MODELS = {m for m in DINO_MODELS} | NEW_MODELS

PRED_COLS = ["set_type", "Complete agreement", "PredictedClass"]


def family_of(name):
    if name.startswith("dinov2_vits14_reg_continued"):
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


def load_pred(root, model, iterdir):
    df = pd.read_json(root / model / iterdir / "predict.json")
    df = df[(df["set_type"] == "Test") & (~df["Complete agreement"].isnull())].copy()
    return df.reset_index(drop=False)[PRED_COLS + ["index"]]


def bootstrap_values(root, model, iterdirs, df_bootstrap):
    """Pooled per-resample macro-F1 across all seeds for one (model, variant)."""
    out = []
    for iterdir in iterdirs:
        df_model = load_pred(root, model, iterdir)
        by_index = df_model.set_index("index")
        for _, row in df_bootstrap.iterrows():
            sel = by_index.loc[by_index.index.intersection(row["test_index"])]
            out.append(f1_score(sel["Complete agreement"].astype(np.int64),
                                sel["PredictedClass"].astype(np.int64),
                                average="macro", zero_division=0) * 100)
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_bootstrap = pd.read_json(BOOTSTRAP_PATH)

    raw, summary = [], []
    for name, iterdirs in ALL_ENTRIES:
        entry = {"model": name, "family": family_of(name), "n_seeds": len(iterdirs)}
        for variant, root in ROOTS.items():
            missing = [d for d in iterdirs if not (root / name / d / "predict.json").exists()]
            if missing:
                print(f"SKIP {name} [{variant}] -- missing {missing}")
                entry[f"{variant}_mean"] = entry[f"{variant}_margin"] = float("nan")
                continue
            vals = bootstrap_values(root, name, iterdirs, df_bootstrap)
            for v in vals:
                raw.append({"model": name, "variant": variant, "macro_f1": v})
            m, mar = mean_and_margin(vals)
            entry[f"{variant}_mean"] = round(m, 2)
            entry[f"{variant}_margin"] = round(mar, 2)
        if {"frozen_mean", "finetuned_mean"} <= entry.keys():
            entry["finetune_gain"] = round(entry["finetuned_mean"] - entry["frozen_mean"], 2)
        summary.append(entry)

    pd.DataFrame(raw).to_csv(RESULTS_DIR / "continued_bootstrap_metrics.csv", index=False)
    df_all = pd.DataFrame(summary)

    df_all.sort_values("finetuned_mean", ascending=False).to_csv(
        RESULTS_DIR / "table_A_all_models.csv", index=False)
    df_all[df_all["model"].isin(SSL_MODELS)].sort_values("finetuned_mean", ascending=False).to_csv(
        RESULTS_DIR / "table_B_ssl_only.csv", index=False)

    print("\n=== TABLE A (all models) ===")
    print(df_all.sort_values("finetuned_mean", ascending=False).to_string(index=False))
    print("\n=== TABLE B (SSL only) ===")
    print(df_all[df_all["model"].isin(SSL_MODELS)].sort_values("finetuned_mean", ascending=False).to_string(index=False))

    df_raw = pd.DataFrame(raw)
    for variant in ("frozen", "finetuned"):
        sub = df_all.dropna(subset=[f"{variant}_mean"]).sort_values(f"{variant}_mean", ascending=False)
        order = sub["model"].tolist()
        fig, ax = plt.subplots(figsize=(20, 8))
        for i, model in enumerate(order):
            vals = df_raw[(df_raw["model"] == model) & (df_raw["variant"] == variant)]["macro_f1"].values
            highlight = model in NEW_MODELS
            color = "#D62728" if model == "dinov2_vits14_reg_continued" else ("#2CA02C" if model == "dinov2_vits14_reg_generic" else "#888888")
            ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=("#ffe0e0" if highlight else "none"), edgecolor=color),
                       medianprops=dict(color=color))
            ax.scatter(np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)), vals,
                       s=4, color=color, alpha=0.5)
        ax.set_xticks(range(len(order)))
        tick_labels = ax.set_xticklabels(order, rotation=60, ha="right")
        for lbl, m in zip(tick_labels, order):
            if m in NEW_MODELS:
                lbl.set_fontweight("bold")
        ax.set_ylabel("Macro F1-score (%)")
        ax.set_title(f"GastroHUN Scenario A, {variant} -- all models ranked "
                     f"(continued-DINOv2 in red, generic dinov2_vits14_reg in green)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"continued_{variant}.png", dpi=200)
        print(f"saved {RESULTS_DIR / f'continued_{variant}.png'}")


if __name__ == "__main__":
    main()
