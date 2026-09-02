#!/bin/bash
#SBATCH --job-name=rangedit-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --exclude=cn01
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-ft-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-ft-%j.out

set -euo pipefail

cd /csehome/p24cs0005/rangediffseg

FINETUNE_FROM=${FINETUNE_FROM:-/scratch/p24cs0005/exp/ckpt/rangediffseg/log_rangedit_semantickitti_wandb_20260826_214845/checkpoint/best_mean_iou_model.pth}
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_semantickitti_finetune_lr5e5_overlap}
LR=${LR:-5e-5}
BATCH_SIZE=${BATCH_SIZE:-6}
N_EPOCHS=${N_EPOCHS:-80}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
VAL_FREQUENCY=${VAL_FREQUENCY:-2}
WINDOW_STRIDE=${WINDOW_STRIDE:-256}
WANDB_ENTITY=${WANDB_ENTITY:-shuhul}
WANDB_MODE=${WANDB_MODE:-online}

echo "job_id=${SLURM_JOB_ID:-none}"
echo "hostname=$(hostname)"
echo "workdir=$(pwd)"
echo "conda_env=difforecast"
echo "finetune_from=${FINETUNE_FROM}"
echo "data_root=${DATA_ROOT}"
echo "save_path=${SAVE_PATH}"
echo "run_id=${RUN_ID}"
echo "lr=${LR}"
echo "batch_size=${BATCH_SIZE}"
echo "n_epochs=${N_EPOCHS}"
echo "warmup_epochs=${WARMUP_EPOCHS}"
echo "val_frequency=${VAL_FREQUENCY}"
echo "window_stride=${WINDOW_STRIDE}"
echo "wandb_mode=${WANDB_MODE}"
echo "start_time=$(date)"

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
  --finetune_from "${FINETUNE_FROM}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --n_epochs "${N_EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --val_frequency "${VAL_FREQUENCY}" \
  --window_stride "${WINDOW_STRIDE}" \
  --use_wandb \
  --wandb_project rangediffseg \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_name "${RUN_ID}" \
  --wandb_mode "${WANDB_MODE}"
