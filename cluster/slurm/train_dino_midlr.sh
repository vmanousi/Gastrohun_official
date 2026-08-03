#!/bin/bash
#SBATCH --job-name=gastrohun_dino_midlr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/train_dino_midlr_%A_%a.out
#SBATCH --error=logs/train_dino_midlr_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# DINO v1 only. We've tested the two extremes: single_lr (backbone at 0.0007,
# same speed as head -- DINO v1's best result, 82-84%) and discriminative_lr
# (backbone at 0.00001, ~70x slower -- hurt 3/4 variants, 77-79%). This tests
# the untested middle: backbone at 0.0001, a gentler 10x-slower-than-head
# ratio, to see whether DINO v1 has its own optimum between the two extremes
# or whether full speed (single_lr) genuinely is its best setting.
MODELS=(dino_vits16 dino_vits8 dino_vitb16 dino_vitb8)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd ~/Gastrohun_official/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_dino_midlr/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, unfrozen_layers: 40, backbone_lr: 0.0001, head_lr: 0.0007, seed: 42, host: $(hostname)"

python train_image_classification.py \
  --model "$MODEL" \
  --input_size 224 \
  --nb_classes 23 \
  --num_epochs_warmup 10 \
  --num_epochs_finetuning 100 \
  --early_stopping 10 \
  --lr_warmup 0.001 \
  --lr_finetuning 0.0007 \
  --backbone_lr_finetuning 0.0001 \
  --seed 42 \
  --gamma_finetuning 0.3 \
  --step_size_finetuning 5 \
  --unfrozen_layers 40 \
  --num_workers 4 \
  --batch_size 40 \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "Train exit code: $?"
ls -la "$OUTPUT_DIR"
