#!/bin/bash
#SBATCH --job-name=gastrohun_optuna_seedchk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/optuna_seedchk_train_%A_%a.out
#SBATCH --error=logs/optuna_seedchk_train_%A_%a.err

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

# Scoped-down seed-robustness check for the two Optuna-confirmed results
# that came out WORSE than their previous best-known recipe (vit_b_16,
# dino_vits16) -- is that a real effect, or single-seed noise? A full,
# independent run (fresh warm-up too, not --resume_from_warmup) with a
# second seed, matching the same methodology used for the earlier DINOv2
# discriminative-LR seed check. Not re-checking the DINOv2 wins here --
# their margins (+2.4 to +3.2 over previous-best, +6 over tuned ViT) are
# large enough that a seed check is low-value there.
MODELS=(vit_b_16 dino_vits16)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=123

case "$MODEL" in
  vit_b_16)
    LR_FINETUNING=0.0004914978622484671
    BACKBONE_LR=0.0004062670552426982
    ;;
  dino_vits16)
    LR_FINETUNING=0.0003574712922600243
    BACKBONE_LR=0.0002543220932812933
    ;;
esac

cd ~/Gastrohun_official_optuna/image_classification/scripts

DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official_optuna/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official_optuna/image_classification/output/Complete_agreement_40_optuna_confirmed_seed${SEED}/${MODEL}/iter1

mkdir -p "$OUTPUT_DIR"
echo "Model: $MODEL, seed: $SEED, lr_finetuning: $LR_FINETUNING, backbone_lr_finetuning: $BACKBONE_LR, host: $(hostname)"

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
  --seed "$SEED" \
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
