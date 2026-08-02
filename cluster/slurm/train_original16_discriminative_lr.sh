#!/bin/bash
#SBATCH --job-name=gastrohun_orig16_disclr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/train_orig16_disclr_%A_%a.out
#SBATCH --error=logs/train_orig16_disclr_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# Control experiment: does discriminative LR help the 16 baseline models too,
# or is it specific to DINOv2? Same recipe as train_original16_repro.sh
# (unfrozen=40, dataset normalization, batch 40) plus --backbone_lr_finetuning
# and --seed, exactly like train_dino_array_discriminative_lr.sh.
MODELS=(convnext_tiny convnext_small convnext_base convnext_large \
        resnet18 resnet34 resnet50 resnet101 resnet152 \
        vgg11 vgg13 vgg16 \
        vit_b_16 vit_b_32 vit_l_16 vit_l_32)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd ~/Gastrohun_official/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_discriminative_lr/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, unfrozen_layers: 40, backbone_lr: 0.00001, head_lr: 0.0007, seed: 42, host: $(hostname)"

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
  --batch_size 40 \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "Train exit code: $?"
ls -la "$OUTPUT_DIR"
