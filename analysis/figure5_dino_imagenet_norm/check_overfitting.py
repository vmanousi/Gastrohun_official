import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plots warmup+finetuning train/val accuracy and macro-F1 from a run's
# history.xlsx, to check whether a low-ranking model (e.g. dinov2_vitl14)
# genuinely underperforms or is just overfitting the 40%-unfrozen fine-tune
# phase. Loss isn't plotted: train_image_classification.py sets logger=False
# on both Lightning Trainer phases, so per-step/per-epoch loss is never
# persisted to history.xlsx or anywhere else -- only metrics_to_dataframe()'s
# train_acc/val_acc/train_f1_macro/val_f1_macro survive.
#
# Usage: python check_overfitting.py <path/to/history.xlsx> [<path/to/another/history.xlsx> ...]
# Each path's parent-of-parent directory name is used as the model label.

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def load_run(path):
    path = Path(path)
    label = path.parent.parent.name  # .../<model>/iter1/history.xlsx
    warmup_train = pd.read_excel(path, sheet_name="wamup_train")
    warmup_val = pd.read_excel(path, sheet_name="wamup_val")
    ft_train = pd.read_excel(path, sheet_name="finetuning_train")
    ft_val = pd.read_excel(path, sheet_name="finetuning_val")
    return label, warmup_train, warmup_val, ft_train, ft_val


def main(paths):
    runs = [load_run(p) for p in paths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    f1_gap_ax, acc_ax = axes

    for label, warmup_train, warmup_val, ft_train, ft_val in runs:
        n_warmup = len(warmup_train)
        train_epochs = range(n_warmup, n_warmup + len(ft_train))
        val_epochs = range(n_warmup, n_warmup + len(ft_val))

        f1_gap_ax.plot(train_epochs, ft_train["train_f1_macro"], label=f"{label} train", linestyle="--")
        f1_gap_ax.plot(val_epochs, ft_val["val_f1_macro"], label=f"{label} val")

        acc_ax.plot(train_epochs, ft_train["train_acc"], label=f"{label} train", linestyle="--")
        acc_ax.plot(val_epochs, ft_val["val_acc"], label=f"{label} val")

    f1_gap_ax.axvline(runs[0][1].shape[0], color="gray", linestyle=":", label="warmup->finetune")
    f1_gap_ax.set_xlabel("epoch")
    f1_gap_ax.set_ylabel("macro F1")
    f1_gap_ax.set_title("Fine-tuning phase: train vs val macro F1\n(widening gap = overfitting)")
    f1_gap_ax.legend(fontsize=7)

    acc_ax.axvline(runs[0][1].shape[0], color="gray", linestyle=":")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_title("Fine-tuning phase: train vs val accuracy")
    acc_ax.legend(fontsize=7)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "overfitting_check.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_overfitting.py <history.xlsx> [<history.xlsx> ...]")
        sys.exit(1)
    main(sys.argv[1:])
