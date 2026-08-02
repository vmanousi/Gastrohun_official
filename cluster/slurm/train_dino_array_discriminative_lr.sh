#!/bin/bash
#SBATCH --job-name=gastrohun_dino_disclr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/train_dino_disclr_%A_%a.out
#SBATCH --error=logs/train_dino_disclr_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# Discriminative-LR track: identical to train_dino_array_repro.sh (unfrozen=40,
# dataset normalization, same batch sizes, same epochs/early-stopping) except
# for --backbone_lr_finetuning (new, small LR just for the unfrozen backbone
# layers) and --seed (fixed, since this is a comparative recipe experiment
# where run-to-run noise matters more than in the fair-repro milestone).
# backbone_lr/head_lr values (1e-5 / 7e-4) are borrowed from a validated
# working recipe in a separate project (FINAL/gastrodino_thesis), which used
# a ~100x backbone/head ratio successfully for DINOv2 fine-tuning.
MODELS=(dino_vits16 dino_vits8 dino_vitb16 dino_vitb8 dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

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
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_discriminative_lr/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, unfrozen_layers: 40, batch_size: $BATCH_SIZE, backbone_lr: 0.00001, head_lr: 0.0007, seed: 42, host: $(hostname)"

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
  --seed 42 \
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
