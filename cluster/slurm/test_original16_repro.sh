#!/bin/bash
#SBATCH --job-name=gastrohun_orig16_repro_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-15
#SBATCH --output=logs/test_orig16_repro_%A_%a.out
#SBATCH --error=logs/test_orig16_repro_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

MODELS=(convnext_tiny convnext_small convnext_base convnext_large \
        resnet18 resnet34 resnet50 resnet101 resnet152 \
        vgg11 vgg13 vgg16 \
        vit_b_16 vit_b_32 vit_l_16 vit_l_32)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Must match the UNFROZEN value used in train_original16_repro.sh -- only
# determines the output dir path here, not the eval logic itself.
UNFROZEN=60

cd ~/Gastrohun_official/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_repro_unfrozen${UNFROZEN}/${MODEL}/iter1

echo "Model: $MODEL, unfrozen_layers: $UNFROZEN, host: $(hostname)"

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
