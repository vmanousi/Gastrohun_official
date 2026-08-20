#!/bin/bash
#SBATCH --job-name=gastrohun_optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/optuna_search_%A_%a.out
#SBATCH --error=logs/optuna_search_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# Isolated Optuna hyperparameter search: 1 supervised ViT + 2 DINO v1 +
# 2 DINOv2, chosen to give both "sides" of the ViT-vs-DINO/DINOv2 comparison
# equal tuning effort (see dino-optuna-search branch). dinov2_vitg14 is
# deliberately excluded from the search itself (~65x the compute of the
# cheapest model here) -- it gets a single confirmation run afterward using
# whatever ratio pattern wins for the other DINOv2 sizes, not a full search.
#
# 24h wall-time is a conservative guess for 20 trials with MedianPruner --
# revisit after the first job's actual per-trial timing is known; the study
# resumes cleanly if interrupted since the warm-up checkpoint is cached and
# Optuna trials are independent (a rerun just does fewer of the n_trials).
MODELS=(vit_b_16 dino_vits16 dino_vitb8 dinov2_vits14 dinov2_vitl14)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd ~/Gastrohun_official_optuna/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official_optuna/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official_optuna/image_classification/output/Optuna_search/${MODEL}

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, n_trials=20, search_epochs=40, search_patience=5, host: $(hostname)"

python optuna_search.py \
  --model "$MODEL" \
  --input_size 224 \
  --nb_classes 23 \
  --unfrozen_layers 40 \
  --num_epochs_warmup 10 \
  --lr_warmup 0.001 \
  --gamma_finetuning 0.3 \
  --step_size_finetuning 5 \
  --n_trials 20 \
  --search_epochs 40 \
  --search_patience 5 \
  --lr_min 0.0001 \
  --lr_max 0.003 \
  --ratio_min 0.001 \
  --ratio_max 1.0 \
  --seed 42 \
  --num_workers 4 \
  --batch_size 40 \
  --data_path "$DATA_PATH" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement" \
  --normalization dataset \
  --output_dir "$OUTPUT_DIR"

echo "Optuna search exit code: $?"
ls -la "$OUTPUT_DIR"
