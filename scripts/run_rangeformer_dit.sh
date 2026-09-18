#!/bin/bash
# Sibling of run_rangediffseg_finetune.sh for the rangeformer_dit_seg branch.
# The baseline submission scripts are left untouched.
#SBATCH --job-name=rfdit-seg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --exclude=cn01
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rfdit-seg-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rfdit-seg-%j.out

set -euo pipefail

cd /csehome/p24cs0005/rangediffseg-rfdit

# ─── knobs ───────────────────────────────────────────────────────────────
DATASET_PATH=${DATASET_PATH:-/scratch/p24cs0005/kitti/dataset}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
CONFIG=${CONFIG:-config_rangeformer_dit.yaml}
RUN_NAME=${RUN_NAME:-rangeformer_dit_seg}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
NUM_GPUS=${NUM_GPUS:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
WANDB_ENTITY=${WANDB_ENTITY:-shuhul}
WANDB_MODE=${WANDB_MODE:-online}

EFFECTIVE_BATCH=$((MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS))

echo "job_id=${SLURM_JOB_ID:-none}"
echo "hostname=$(hostname)"
echo "workdir=$(pwd)"
echo "conda_env=difforecast"
echo "config=${CONFIG}"
echo "dataset_path=${DATASET_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "run_name=${RUN_NAME}"
echo "learning_rate=${LEARNING_RATE}"
echo "---------------- batch composition ----------------"
echo "micro batch size     = ${MICRO_BATCH_SIZE}"
echo "number of GPUs       = ${NUM_GPUS}"
echo "accumulation steps   = ${GRAD_ACCUM_STEPS}"
echo "effective batch size = ${EFFECTIVE_BATCH}   (${MICRO_BATCH_SIZE} x ${GRAD_ACCUM_STEPS} x ${NUM_GPUS})"
echo "---------------------------------------------------"
echo "start_time=$(date)"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast

nvidia-smi || true

# The repository launches training with plain `python train.py`; that mechanism
# is preserved rather than adding a competing distributed launcher.
python -u train.py "${CONFIG}" \
  --data_root "${DATASET_PATH}" \
  --save_path "${OUTPUT_DIR}" \
  --id "${RUN_NAME}" \
  --model_type rangeformer_dit_seg \
  --batch_size "${MICRO_BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --use_wandb \
  --wandb_project rangediffseg \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_name "${RUN_NAME}" \
  --wandb_mode "${WANDB_MODE}"

echo "end_time=$(date)"
