#!/bin/bash
#SBATCH --job-name=gastrohun_dino_imgnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/train_imagenet_norm_%A_%a.out
#SBATCH --error=logs/train_imagenet_norm_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

MODELS=(dino_vits16 dino_vits8 dino_vitb16 dino_vitb8 dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

case "$MODEL" in
  dinov2_vitl14|dinov2_vitg14)
    BATCH_SIZE=16
    ;;
  *)
    BATCH_SIZE=40
    ;;
esac

cd ~/Gastrohun_official/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_imagenet_norm/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, batch size: $BATCH_SIZE, normalization: imagenet, host: $(hostname)"

python train_image_classification.py \
  --model "$MODEL" \
  --input_size 224 \
  --nb_classes 23 \
  --num_epochs_warmup 10 \
  --num_epochs_finetuning 100 \
  --early_stopping 10 \
  --lr_warmup 0.001 \
  --lr_finetuning 0.0007 \
  --gamma_finetuning 0.3 \
  --step_size_finetuning 5 \
  --unfrozen_layers 40 \
  --num_workers 4 \
  --batch_size "$BATCH_SIZE" \
  --normalization imagenet \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "Train exit code: $?"
ls -la "$OUTPUT_DIR"
