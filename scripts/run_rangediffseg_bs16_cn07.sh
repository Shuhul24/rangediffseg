#!/bin/bash
#SBATCH --job-name=rangedit-bs16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-bs16-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-bs16-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

# Companion to the batch-4 run: same architecture and data pipeline, but at
# batch 16, which measures ~2x faster in this pipeline (22.4 vs 12.0 scans/s)
# because the fixed per-iteration overhead amortises over 4x fewer steps.
# Peak memory is 41.6 GiB of the card's 95, so it runs on a second cn07 GPU
# alongside the batch-4 job.
#
# Hyperparameters changed for the 4x batch, and why:
#   lr 2e-4 -> 4e-4   sqrt scaling. Linear scaling (8e-4) is usually too
#                     aggressive for AdamW. This is a convention, not a
#                     measured optimum -- it is the main risk in this run.
#   warmup 2 -> 3     an epoch is now 1195 iterations instead of 4782, so
#                     warmup measured in epochs buys 4x fewer steps; 3 epochs
#                     restores a safer fraction given the doubled lr.
#   workers 8 -> 12   4x the samples per iteration to project and augment.
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_scratch_bs16_lr4e4_cn07}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-4e-4}
N_EPOCHS=${N_EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
VAL_FREQUENCY=${VAL_FREQUENCY:-2}
WANDB_ENTITY=${WANDB_ENTITY:-shuhul}
WANDB_MODE=${WANDB_MODE:-online}

echo "job_id=${SLURM_JOB_ID:-none}  host=$(hostname)  start=$(date)"
echo "run_id=${RUN_ID}  batch=${BATCH_SIZE}  lr=${LR}  epochs=${N_EPOCHS}"
echo "adapter+fusion+RangeAug from config.yaml; single GPU, no DDP"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u train.py --config config.yaml \
  --data_root "${DATA_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --id "${RUN_ID}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --n_epochs "${N_EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --val_frequency "${VAL_FREQUENCY}" \
  --num_workers 12 \
  --use_wandb \
  --wandb_project rangediffseg \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_name "${RUN_ID}" \
  --wandb_mode "${WANDB_MODE}"

echo "end=$(date)"
