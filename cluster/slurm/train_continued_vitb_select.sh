#!/bin/bash
#SBATCH --job-name=gastrohun_vitb_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/vitb_select_%A_%a.out
#SBATCH --error=logs/vitb_select_%A_%a.err

# Project 3 (ViT-B/14+reg4) stage 2: run the top-3 continued-SSL teacher
# checkpoints (from the frozen linear-probe sweep, outputs/downstream_vitb/sweep)
# through the real gastrohun pipeline, 1 seed, both scenarios (unfrozen 0 and 40).
# Pick the winner by validation macro-F1 with analysis/continued_dinov2_vitb/pick_checkpoint.py.
#
#   CANDS="89999 99999 49999" sbatch cluster/slurm/train_continued_vitb_select.sh
#
# array index = cand_idx * 2 + unfrozen_idx   (3 cands x 2 scenarios = 6)

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

: "${CANDS:?set CANDS to the 3 top iter numbers, e.g. CANDS=\"89999 99999 49999\"}"
read -ra CAND_ARR <<< "$CANDS"
UNFROZENS=(0 40)

ci=$(( SLURM_ARRAY_TASK_ID / 2 ))
ui=$(( SLURM_ARRAY_TASK_ID % 2 ))
CAND=${CAND_ARR[$ci]}
UNFROZEN=${UNFROZENS[$ui]}

SSL_REPO=~/continue_ssl_pretrain_dinov2
CKPT="$SSL_REPO/outputs/full_run_vitb/eval/training_${CAND}/teacher_checkpoint.pth"
[ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 1; }

cd ~/Gastrohun_official/image_classification/scripts
DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_repro_vitb_unfrozen${UNFROZEN}/dinov2_vitb14_reg_cand${CAND}/iter1
mkdir -p "$OUTPUT_DIR"

echo "cand iter $CAND, unfrozen $UNFROZEN, ckpt $CKPT, host $(hostname)"

python train_image_classification.py \
  --model dinov2_vitb14_reg \
  --pretrained_backbone "$CKPT" \
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
