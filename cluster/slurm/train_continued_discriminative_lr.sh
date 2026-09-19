#!/bin/bash
#SBATCH --job-name=gastrohun_cont_disclr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/train_cont_disclr_%A_%a.out
#SBATCH --error=logs/train_cont_disclr_%A_%a.err

# EXPLORATORY / secondary track -- "what's each architecture's ceiling", not
# the controlled comparison. Fully isolated from everything else in this repo:
#   - own output root (Complete_agreement_40_discriminative_lr_continued/),
#     distinct from Complete_agreement_40_discriminative_lr/ used by the
#     original 4 DINOv2 baselines -- no shared paths, nothing here can
#     collide with or be skipped-because-exists by any prior run.
#   - own script file -- does not touch train_continued_reg.sh or any
#     existing cluster/slurm/*.sh.
# Recipe is an exact copy of train_dino_array_discriminative_lr.sh (the one
# that fixed DINOv2's "recipe-driven underperformance", +7 to +9.4 macro F1
# over single_lr for the original 4 DINOv2 variants): backbone_lr=1e-5,
# head_lr=7e-4, seed=42 fixed (matching that track's convention -- 1 seed,
# not 3, so our two new entries are directly comparable to the existing ones
# in the same secondary table).
#
#   WINNER_ITER=69999 sbatch cluster/slurm/train_continued_discriminative_lr.sh

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

: "${WINNER_ITER:?set WINNER_ITER, e.g. WINNER_ITER=69999}"

MODEL_KEYS=(generic continued)
MODEL_KEY=${MODEL_KEYS[$SLURM_ARRAY_TASK_ID]}

SSL_REPO=~/continue_ssl_pretrain_dinov2
case "$MODEL_KEY" in
  generic)   BACKBONE="$SSL_REPO/checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth" ;;
  continued) BACKBONE="$SSL_REPO/outputs/full_run/eval/training_${WINNER_ITER}/teacher_checkpoint.pth" ;;
esac
[ -f "$BACKBONE" ] || { echo "MISSING $BACKBONE"; exit 1; }

cd ~/Gastrohun_official/image_classification/scripts
DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_discriminative_lr_continued/dinov2_vits14_reg_${MODEL_KEY}/iter1
mkdir -p "$OUTPUT_DIR"

echo "model_key $MODEL_KEY, backbone_lr 1e-5, head_lr 7e-4, seed 42, backbone $BACKBONE, host $(hostname)"

python train_image_classification.py \
  --model dinov2_vits14_reg \
  --pretrained_backbone "$BACKBONE" \
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

echo "exit $?"; ls -la "$OUTPUT_DIR"
