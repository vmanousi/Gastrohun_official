#!/bin/bash
#SBATCH --job-name=gastrohun_dino_midlr_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/test_dino_midlr_%A_%a.out
#SBATCH --error=logs/test_dino_midlr_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

MODELS=(dino_vits16 dino_vits8 dino_vitb16 dino_vitb8)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd ~/Gastrohun_official/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_dino_midlr/${MODEL}/iter1

echo "Model: $MODEL, host: $(hostname)"

python test_image_classification.py \
  --model "$MODEL" \
  --input_size 224 \
  --nb_classes 23 \
  --num_workers 4 \
  --batch_size 40 \
  --model_path "$OUTPUT_DIR/best-model-val_f1_macro.ckpt" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "Test exit code: $?"
ls -la "$OUTPUT_DIR"
