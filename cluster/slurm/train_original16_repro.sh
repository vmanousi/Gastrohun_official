#!/bin/bash
#SBATCH --job-name=gastrohun_orig16_repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/train_orig16_repro_%A_%a.out
#SBATCH --error=logs/train_orig16_repro_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# The 16 models the paper itself trained (Scenario A, complete-agreement labels).
MODELS=(convnext_tiny convnext_small convnext_base convnext_large \
        resnet18 resnet34 resnet50 resnet101 resnet152 \
        vgg11 vgg13 vgg16 \
        vit_b_16 vit_b_32 vit_l_16 vit_l_32)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# IMAGECLASSIFICATION.md's own shipped example uses 60 -- the paper's methods
# text says 40, and the code's argparse default is 10, with an unexplained
# "#40" comment next to it. All three values trace back to the original
# authors with no reconciliation. Starting with 60 since it's the only one
# backed by a concrete, runnable command they actually shipped; change this
# and the output dir suffix together to test 40 or 10 instead.
UNFROZEN=60

cd ~/Gastrohun_official/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_repro_unfrozen${UNFROZEN}/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, unfrozen_layers: $UNFROZEN, host: $(hostname)"

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
  --batch_size 40 \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "Train exit code: $?"
ls -la "$OUTPUT_DIR"
