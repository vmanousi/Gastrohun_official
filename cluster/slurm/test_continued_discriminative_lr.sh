#!/bin/bash
#SBATCH --job-name=gastrohun_cont_disclr_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/test_cont_disclr_%A_%a.out
#SBATCH --error=logs/test_cont_disclr_%A_%a.err

# Tests the 2 classifiers trained by train_continued_discriminative_lr.sh.
# Same isolated output root, same array layout.
#   WINNER_ITER=69999 sbatch cluster/slurm/test_continued_discriminative_lr.sh

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

: "${WINNER_ITER:?set WINNER_ITER (same value used for training)}"

MODEL_KEYS=(generic continued)
MODEL_KEY=${MODEL_KEYS[$SLURM_ARRAY_TASK_ID]}

SSL_REPO=~/continue_ssl_pretrain_dinov2
case "$MODEL_KEY" in
  generic)   BACKBONE="$SSL_REPO/checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth" ;;
  continued) BACKBONE="$SSL_REPO/outputs/full_run/eval/training_${WINNER_ITER}/teacher_checkpoint.pth" ;;
esac

cd ~/Gastrohun_official/image_classification/scripts
DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_discriminative_lr_continued/dinov2_vits14_reg_${MODEL_KEY}/iter1

echo "model_key $MODEL_KEY, host $(hostname)"

python test_image_classification.py \
  --model dinov2_vits14_reg \
  --pretrained_backbone "$BACKBONE" \
  --input_size 224 \
  --nb_classes 23 \
  --num_workers 4 \
  --batch_size 40 \
  --model_path "$OUTPUT_DIR/best-model-val_f1_macro.ckpt" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement"

echo "exit $?"; ls -la "$OUTPUT_DIR"
