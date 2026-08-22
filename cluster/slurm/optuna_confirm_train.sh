#!/bin/bash
#SBATCH --job-name=gastrohun_optuna_confirm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/optuna_confirm_train_%A_%a.out
#SBATCH --error=logs/optuna_confirm_train_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# Full-budget confirmation runs using each model's Optuna-winning
# (lr_finetuning, backbone_lr_finetuning), evaluated the same way as every
# other experiment in this project (100 epochs, patience 10, test-set
# bootstrap comparison to follow). 6 models total; ampere-extd's
# MaxSubmitPU=4 means submitting all at once will fail -- submit array
# 0-3 first, then 4-5 once a slot frees up:
#   sbatch --array=4-5 cluster/slurm/optuna_confirm_train.sh
#
# dinov2_vitg14 was NOT part of the Optuna search itself (too expensive --
# see optuna_search.sh) -- its (lr_finetuning, backbone_lr_finetuning)
# below are extrapolated from dinov2_vits14's and dinov2_vitl14's winning
# ratios (0.063 and 0.035) via a log-linear fit of ratio vs. parameter
# count, giving ratio=0.026, not a searched value. Flagged as an estimate,
# not a finding, in any resulting analysis.
MODELS=(vit_b_16 dino_vits16 dino_vitb8 dinov2_vits14 dinov2_vitl14 dinov2_vitg14)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

case "$MODEL" in
  vit_b_16)
    LR_FINETUNING=0.0004914978622484671
    BACKBONE_LR=0.0004062670552426982
    ;;
  dino_vits16)
    LR_FINETUNING=0.0003574712922600243
    BACKBONE_LR=0.0002543220932812933
    ;;
  dino_vitb8)
    LR_FINETUNING=0.002324515410053115
    BACKBONE_LR=0.00028423572797159927
    ;;
  dinov2_vits14)
    LR_FINETUNING=0.001205712628744377
    BACKBONE_LR=0.0000753736006579975
    ;;
  dinov2_vitl14)
    LR_FINETUNING=0.0011734402754368836
    BACKBONE_LR=0.00004124399243208564
    ;;
  dinov2_vitg14)
    LR_FINETUNING=0.0012
    BACKBONE_LR=0.000031  # extrapolated, see comment above -- not searched
    ;;
esac

# Same batch-size exception as every other track in this project
case "$MODEL" in
  dinov2_vitg14)
    BATCH_SIZE=16
    ;;
  *)
    BATCH_SIZE=40
    ;;
esac

cd ~/Gastrohun_official_optuna/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official_optuna/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official_optuna/image_classification/output/Complete_agreement_40_optuna_confirmed/${MODEL}/iter1
WARMUP_CKPT=~/Gastrohun_official_optuna/image_classification/output/Optuna_search/${MODEL}/warmup/warmup_model.pt

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, lr_finetuning: $LR_FINETUNING, backbone_lr_finetuning: $BACKBONE_LR, batch_size: $BATCH_SIZE, host: $(hostname)"

if [ -f "$WARMUP_CKPT" ]; then
  echo "Reusing cached warm-up checkpoint from the Optuna search: $WARMUP_CKPT"
  RESUME_ARGS="--resume_from_warmup $WARMUP_CKPT"
else
  echo "No cached warm-up checkpoint (expected for dinov2_vitg14, which wasn't part of the search) -- running Phase 1 normally."
  RESUME_ARGS=""
fi

python train_image_classification.py \
  --model "$MODEL" \
  --input_size 224 \
  --nb_classes 23 \
  --num_epochs_warmup 10 \
  --num_epochs_finetuning 100 \
  --early_stopping 10 \
  --lr_warmup 0.001 \
  --lr_finetuning "$LR_FINETUNING" \
  --backbone_lr_finetuning "$BACKBONE_LR" \
  --seed 42 \
  --gamma_finetuning 0.3 \
  --step_size_finetuning 5 \
  --unfrozen_layers 40 \
  --num_workers 4 \
  --batch_size "$BATCH_SIZE" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --official_split "$DATA_SPLIT" \
  --label "Complete agreement" \
  $RESUME_ARGS

echo "Train exit code: $?"
ls -la "$OUTPUT_DIR"
