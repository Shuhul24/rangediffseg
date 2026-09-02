#!/bin/bash
#SBATCH --job-name=rangedit-scratch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-scratch-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-scratch-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

# From scratch: the off-the-shelf DiT weights, no warm start. batch_size and lr
# are deliberately left at the 58.29 baseline's values so the only differences
# are the spatial-prior adapter, multi-level fusion, RangeAug and the
# overlapping eval windows. Throughput on these cards is flat in batch size,
# so a larger batch would cost an unvalidated lr change for no speed gain.
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_scratch_adapter_rangeaug_cn07}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-2e-4}
N_EPOCHS=${N_EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}
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
  --num_workers 8 \
  --use_wandb \
  --wandb_project rangediffseg \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_name "${RUN_ID}" \
  --wandb_mode "${WANDB_MODE}"

echo "end=$(date)"
