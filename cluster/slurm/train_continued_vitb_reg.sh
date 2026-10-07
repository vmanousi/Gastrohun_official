#!/bin/bash
#SBATCH --job-name=gastrohun_vitb_reg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/train_vitb_reg_%A_%a.out
#SBATCH --error=logs/train_vitb_reg_%A_%a.err

# Project 3 (ViT-B/14+reg4) analogue of train_continued_reg.sh -- the final,
# fair-comparison entries:
#   generic dinov2_vitb14_reg  vs  continued (winning teacher checkpoint)
#   x  unfrozen {0, 40}   x  seed {1, 2, 3}     = 12 runs
# Identical recipe to train_continued_reg.sh (same lr_finetuning=0.0007,
# inherited from the ViT-S Optuna search, not re-searched for ViT-B -- see
# cluster/slurm/train_continued_vitb_select.sh header). Only the backbone
# weights and output directory (*_vitb_*) differ, so this never touches
# Project 1's files.
#
#   WINNER_ITER=49999 sbatch cluster/slurm/train_continued_vitb_reg.sh
#
# array index -> (model, unfrozen, seed):
#   seed  = SEEDS[ idx % 3 ]
#   unfrz = UNFROZENS[ (idx / 3) % 2 ]
#   model = MODEL_KEYS[ idx / 6 ]

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

: "${WINNER_ITER:?set WINNER_ITER to the teacher-checkpoint iter chosen in the selection step}"

MODEL_KEYS=(generic continued)
UNFROZENS=(0 40)
SEEDS=(1 2 3)

si=$(( SLURM_ARRAY_TASK_ID % 3 ))
ui=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
mi=$(( SLURM_ARRAY_TASK_ID / 6 ))
MODEL_KEY=${MODEL_KEYS[$mi]}
UNFROZEN=${UNFROZENS[$ui]}
SEED=${SEEDS[$si]}

SSL_REPO=~/continue_ssl_pretrain_dinov2
case "$MODEL_KEY" in
  generic)   BACKBONE="$SSL_REPO/checkpoints/dinov2_vitb14_reg4_pretrain_wrapped_224.pth" ;;
  continued) BACKBONE="$SSL_REPO/outputs/full_run_vitb/eval/training_${WINNER_ITER}/teacher_checkpoint.pth" ;;
esac
[ -f "$BACKBONE" ] || { echo "MISSING $BACKBONE"; exit 1; }

cd ~/Gastrohun_official/image_classification/scripts
DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_repro_vitb_unfrozen${UNFROZEN}/dinov2_vitb14_reg_${MODEL_KEY}/iter${SEED}
mkdir -p "$OUTPUT_DIR"

echo "model_key $MODEL_KEY, unfrozen $UNFROZEN, seed $SEED, backbone $BACKBONE, host $(hostname)"

python train_image_classification.py \
  --model dinov2_vitb14_reg \
  --pretrained_backbone "$BACKBONE" \
  --seed "$SEED" \
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

echo "exit $?"; ls -la "$OUTPUT_DIR"
