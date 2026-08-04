#!/bin/bash
#SBATCH --job-name=gastrohun_dinov2_seedchk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/train_dinov2_seedchk_%A_%a.out
#SBATCH --error=logs/train_dinov2_seedchk_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# Robustness check for the headline discriminative-LR result: is the large
# DINOv2 gain (+7 to +9.4 macro F1 over single-LR) real, or a lucky single
# seed=42 run? Scoped to the smallest (vits14) and largest (vitg14) DINOv2
# variants, bracketing the family. Must set SEED explicitly, e.g.:
#   sbatch --export=ALL,SEED=123 train_dinov2_seed_check.sh
MODELS=(dinov2_vits14 dinov2_vitg14)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEED:?"Must set SEED, e.g. sbatch --export=ALL,SEED=123 train_dinov2_seed_check.sh"}

case "$MODEL" in
  dinov2_vitg14)
    BATCH_SIZE=16
    ;;
  *)
    BATCH_SIZE=40
    ;;
esac

cd ~/Gastrohun_official/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_discriminative_lr_seed${SEED}/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, seed: $SEED, batch_size: $BATCH_SIZE, backbone_lr: 0.00001, head_lr: 0.0007, host: $(hostname)"

python train_image_classification.py \
  --model "$MODEL" \
  --input_size 224 \
  --nb_classes 23 \
  --num_epochs_warmup 10 \
  --num_epochs_finetuning 100 \
  --early_stopping 10 \
  --lr_warmup 0.001 \
  --lr_finetuning 0.0007 \
  --backbone_lr_finetuning 0.00001 \
  --seed "$SEED" \
  --gamma_finetuning 0.3 \
  --step_size_finetuning 5 \
  --unfrozen_layers 40 \
  --num_workers 4 \
  --batch_size "$BATCH_SIZE" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "Train exit code: $?"
ls -la "$OUTPUT_DIR"
