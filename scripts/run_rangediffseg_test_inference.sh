#!/bin/bash
#SBATCH --job-name=rangedit-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --exclude=cn01
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-test-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-test-%j.out

set -euo pipefail

cd /csehome/p24cs0005/rangediffseg

CHECKPOINT=${CHECKPOINT:-/scratch/p24cs0005/exp/ckpt/rangediffseg/log_rangedit_semantickitti_wandb_20260826_214845/checkpoint/best_mean_iou_model.pth}
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_test_best_mean_iou}

echo "job_id=${SLURM_JOB_ID:-none}"
echo "hostname=$(hostname)"
echo "workdir=$(pwd)"
echo "conda_env=difforecast"
echo "checkpoint=${CHECKPOINT}"
echo "data_root=${DATA_ROOT}"
echo "save_path=${SAVE_PATH}"
echo "run_id=${RUN_ID}"
echo "start_time=$(date)"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast

nvidia-smi || true

python -u inference.py --config config.yaml \
  --data_root "${DATA_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --id "${RUN_ID}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --num_workers 4 \
  --save_eval_results
