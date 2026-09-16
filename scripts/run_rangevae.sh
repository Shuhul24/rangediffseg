#!/bin/bash
#SBATCH --job-name=rangevae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangevae-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangevae-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

# Stage 1 of the latent-diffusion route: a RangeLDM-style KL autoencoder that
# compresses 64x2048 range images to a 4x16x512 latent. z_channels=4 is the
# number that matters -- it is what DiT-XL/2's x_embedder expects, so a later
# stage can run the trunk (and a diffusion loss) on its native input.
# Self-supervised, so no labels are used.
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangevae_z4_ch64}
BATCH_SIZE=${BATCH_SIZE:-auto}
EPOCHS=${EPOCHS:-200}
LR=${LR:-1e-4}

echo "job_id=${SLURM_JOB_ID:-none}  host=$(hostname)  start=$(date)"
echo "run_id=${RUN_ID}  batch=${BATCH_SIZE}  epochs=${EPOCHS}  lr=${LR}"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
# Reduce allocator fragmentation; the OOM at batch 32 reported 1.69 GiB reserved
# but unallocated.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# cn07 is heterogeneous: RTX A6000 (48 GB) and RTX PRO 6000 Blackwell (96 GB),
# with no SLURM feature to tell them apart or request one. Detect what we got.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "detected: ${GPU_NAME} (${GPU_MEM} MiB)"

# REQUIRE_GPU=blackwell makes the job exit immediately on the wrong card, so a
# resubmit loop costs ~40 s per attempt instead of a wasted allocation.
if [ -n "${REQUIRE_GPU:-}" ]; then
  case "${REQUIRE_GPU}" in
    blackwell) echo "${GPU_NAME}" | grep -qi "blackwell" || { echo "WRONG_GPU: wanted Blackwell, got ${GPU_NAME}"; exit 1; } ;;
    a6000)     echo "${GPU_NAME}" | grep -qi "a6000"     || { echo "WRONG_GPU: wanted A6000, got ${GPU_NAME}"; exit 1; } ;;
  esac
fi

# Size the batch to the card unless one was given explicitly.
if [ "${BATCH_SIZE}" = "auto" ]; then
  if [ "${GPU_MEM}" -gt 60000 ]; then BATCH_SIZE=32; else BATCH_SIZE=16; fi
  echo "auto batch_size=${BATCH_SIZE}"
fi

python -u train_vae.py \
  --data_root "${DATA_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --id "${RUN_ID}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --num_workers 12 \
  --val_every 2

echo "end=$(date)"
