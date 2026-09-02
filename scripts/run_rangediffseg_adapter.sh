#!/bin/bash
#SBATCH --job-name=rangedit-adapt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-adapt-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-adapt-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

# Trained from scratch (off-the-shelf DiT weights), NOT warm-started, so the
# comparison against the 58.29 baseline isolates multi-level feature fusion.
# Every other hyperparameter is deliberately left at the baseline value.
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_semantickitti_adapter_rangeaug}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-2e-4}
N_EPOCHS=${N_EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}
VAL_FREQUENCY=${VAL_FREQUENCY:-2}
WANDB_ENTITY=${WANDB_ENTITY:-shuhul}
WANDB_MODE=${WANDB_MODE:-online}

echo "job_id=${SLURM_JOB_ID:-none}  host=$(hostname)  start=$(date)"
echo "run_id=${RUN_ID}  batch_size=${BATCH_SIZE}  lr=${LR}  n_epochs=${N_EPOCHS}"
echo "fusion_layers + adapter_layers + range_augmentation = (from config.yaml)"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
nvidia-smi || true

python -u train.py --config config.yaml \
  --data_root "${DATA_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --id "${RUN_ID}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --n_epochs "${N_EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --val_frequency "${VAL_FREQUENCY}" \
  --use_wandb \
  --wandb_project rangediffseg \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_name "${RUN_ID}" \
  --wandb_mode "${WANDB_MODE}"

echo "end=$(date)"
