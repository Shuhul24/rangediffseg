#!/bin/bash
#SBATCH --job-name=rangedit-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-val-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-val-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

CHECKPOINT=${CHECKPOINT:-/scratch/p24cs0005/exp/ckpt/rangediffseg/log_rangedit_semantickitti_finetune_lr5e5_overlap/checkpoint/best_mean_iou_model.pth}
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
FUSION=${FUSION:-none}
ADAPTER=${ADAPTER:-none}

echo "job_id=${SLURM_JOB_ID:-none}  host=$(hostname)  start=$(date)"
echo "checkpoint=${CHECKPOINT}"
echo "fusion_layers=${FUSION}"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
nvidia-smi || true

# Same checkpoint, same windows: the only difference is the back-projection,
# so the two tables isolate exactly what KNN post-processing is worth.
echo "================ WITH KNN post-processing ================"
python -u inference.py --config config.yaml \
  --data_root "${DATA_ROOT}" --save_path "${SAVE_PATH}" \
  --id rangedit_val_knn --checkpoint "${CHECKPOINT}" \
  --split val --num_workers 4 --fusion_layers "${FUSION}" --adapter_layers "${ADAPTER}"

echo "end=$(date)"
