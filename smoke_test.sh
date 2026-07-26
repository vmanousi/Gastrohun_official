#!/bin/bash
#SBATCH --job-name=gastrohun_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err
set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate
cd ~/Gastrohun_official/image_classification/scripts
DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/smoke_test/resnet18/iter1
echo "Host: $(hostname)"
nvidia-smi || true
python train_image_classification.py \
  --model resnet18 \
  --input_size 224 \
  --nb_classes 23 \
  --num_epochs_warmup 1 \
  --num_epochs_finetuning 1 \
  --early_stopping 10 \
  --lr_warmup 0.001 \
  --lr_finetuning 0.0007 \
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