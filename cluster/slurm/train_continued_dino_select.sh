#!/bin/bash
#SBATCH --job-name=gastrohun_dino_cont_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/train_dino_cont_select_%A_%a.out
#SBATCH --error=logs/train_dino_cont_select_%A_%a.err

# DINO v1 analogue of train_continued_select.sh -- run the top-3 continued-SSL
# backbone checkpoints (from the fast frozen-probe sweep in the
# continue_ssl_pretrain_dino_v1 repo) through the REAL gastrohun pipeline,
# 1 seed, both scenarios (unfrozen 0 and 40). Pick the winner by val
# macro-F1, then feed it to train_continued_dino_reg.sh for the 3-seed final run.
#
#   CANDS="checkpoint0051 checkpoint0068 checkpoint0034" sbatch cluster/slurm/train_continued_dino_select.sh
#
# array index = cand_idx * 2 + unfrozen_idx   (3 cands x 2 scenarios = 6)

set -euo pipefail
source ~/diplomatiki2/.venv/bin/activate

: "${CANDS:?set CANDS to the 3 top checkpoint names, e.g. CANDS=\"checkpoint0051 checkpoint0068 checkpoint0034\"}"
read -ra CAND_ARR <<< "$CANDS"
UNFROZENS=(0 40)

ci=$(( SLURM_ARRAY_TASK_ID / 2 ))
ui=$(( SLURM_ARRAY_TASK_ID % 2 ))
CAND=${CAND_ARR[$ci]}
UNFROZEN=${UNFROZENS[$ui]}

SSL_REPO=~/continue_ssl_pretrain_dino_v1
CKPT="$SSL_REPO/outputs/full_run/backbones/${CAND}_backbone.pth"
[ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 1; }

cd ~/Gastrohun_official/image_classification/scripts
DATA_PATH=~/Datasets/GastroHun/Labeled_Images_GastroHun
DATA_SPLIT=~/Gastrohun_official/official_splits/image_classification.csv
OUTPUT_DIR=~/Gastrohun_official/image_classification/output/Complete_agreement_40_repro_dino_unfrozen${UNFROZEN}/dino_vits16_cand${CAND}/iter1
mkdir -p "$OUTPUT_DIR"

echo "cand $CAND, unfrozen $UNFROZEN, ckpt $CKPT, host $(hostname)"

python train_image_classification.py \
  --model dino_vits16 \
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
