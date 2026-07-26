#!/bin/bash
#SBATCH --job-name=gastrohun_test_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=test_smoke_%j.out
#SBATCH --error=test_smoke_%j.err
set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate
cd ~/Gastrohun_official/image_classification/scripts
DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/smoke_test/resnet18/iter1
python test_image_classification.py \
  --model resnet18 \
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