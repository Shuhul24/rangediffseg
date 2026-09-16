#!/bin/bash
#SBATCH --job-name=ditb-diff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/ditb-diff-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/ditb-diff-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

# DiT-B/2 trained FROM SCRATCH with a joint objective:
#   L = focal + Lovasz  (segmentation)  +  lambda * MSE(eps_hat, eps)  (diffusion)
# No pretrained weights exist for DiT-S, so nothing is frozen. A timestep is
# drawn per sample, the input is noised, and the restored noise head predicts
# the epsilon that was added -- so DiT's own objective trains alongside
# segmentation and both gradients pass through the same trunk.
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_b_diffusion}
BATCH_SIZE=${BATCH_SIZE:-auto}

echo "job_id=${SLURM_JOB_ID:-none}  host=$(hostname)  start=$(date)"
echo "run_id=${RUN_ID}  batch=${BATCH_SIZE}  time_limit=5d"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate difforecast
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# cn07 mixes RTX A6000 (48 GB) and RTX PRO 6000 Blackwell (96 GB) with no SLURM
# feature to request one. DiT-B is ~2x DiT-S's width, so size to the card.
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${BATCH_SIZE}" = "auto" ]; then
  if [ "${GPU_MEM}" -gt 60000 ]; then BATCH_SIZE=16; else BATCH_SIZE=8; fi
  echo "auto batch_size=${BATCH_SIZE} (GPU has ${GPU_MEM} MiB)"
fi

python -u train.py config_ditb_diffusion.yaml \
  --data_root "${DATA_ROOT}" --save_path "${SAVE_PATH}" --id "${RUN_ID}" \
  --batch_size "${BATCH_SIZE}" --num_workers 12 \
  --use_wandb --wandb_project rangediffseg --wandb_entity shuhul \
  --wandb_name "${RUN_ID}" --wandb_mode online

echo "end=$(date)"
