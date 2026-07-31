#!/bin/bash
#SBATCH --job-name=gastrohun_dino_repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/train_dino_repro_%A_%a.out
#SBATCH --error=logs/train_dino_repro_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# Full-parity DINO/DINOv2 track: same recipe as train_original16_repro.sh
# (dataset normalization, uniform batch size, same everything else) -- NOT
# the imagenet-norm track (train_dino_array_imagenet_norm.sh), which stays
# untouched as a separate follow-up experiment. This is what actually goes
# into Figure 5 alongside the baselines for the fair-comparison milestone.
MODELS=(dino_vits16 dino_vits8 dino_vitb16 dino_vitb8 dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Must match the UNFROZEN value used for the baseline reproduction that won
# (see analysis/original16_repro/compare_repro.py). Override with
# `sbatch --export=ALL,UNFROZEN=40 train_dino_array_repro.sh`.
UNFROZEN=${UNFROZEN:-60}

# dinov2_vitg14 (1.1B params, by far the largest model in this comparison --
# the next biggest, ConvNeXt_Large, is ~200M) hit a genuine CUDA OOM at
# batch_size=40/unfrozen=40 on a 40GB A100. Every other model, including
# dinov2_vitl14 (304M), trains fine at 40. Documented single-model exception,
# not a methodology choice.
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
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_repro_unfrozen${UNFROZEN}/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, unfrozen_layers: $UNFROZEN, batch_size: $BATCH_SIZE, host: $(hostname)"

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
  --unfrozen_layers "$UNFROZEN" \
  --num_workers 4 \
  --batch_size "$BATCH_SIZE" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "Train exit code: $?"
ls -la "$OUTPUT_DIR"
