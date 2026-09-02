#!/bin/bash
#SBATCH --job-name=rangedit-ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-ddp-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-ddp-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

# cn07 carries RTX PRO 6000 Blackwell cards (~96 GiB each). Measured peak
# memory for this architecture is ~41.6 GiB at batch 16, so 16 per GPU leaves
# comfortable headroom. Throughput is flat in batch size on these cards, so the
# larger batch buys optimisation quality (BatchNorm statistics, gradient noise)
# rather than speed; the speed comes from the second GPU.
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_semantickitti_ddp}
NPROC=${NPROC:-2}
BATCH_SIZE=${BATCH_SIZE:-16}          # per GPU -> 32 effective at NPROC=2
LR=${LR:-5.7e-4}                      # sqrt scaling of 2e-4 for an 8x batch
N_EPOCHS=${N_EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}
VAL_FREQUENCY=${VAL_FREQUENCY:-2}
WANDB_ENTITY=${WANDB_ENTITY:-shuhul}
WANDB_MODE=${WANDB_MODE:-online}

echo "job_id=${SLURM_JOB_ID:-none}  host=$(hostname)  start=$(date)"
echo "nproc=${NPROC}  batch_per_gpu=${BATCH_SIZE}  effective=$((BATCH_SIZE*NPROC))  lr=${LR}"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

export OMP_NUM_THREADS=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + RANDOM % 20000))

torchrun --standalone --nproc_per_node="${NPROC}" train.py --config config.yaml \
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
