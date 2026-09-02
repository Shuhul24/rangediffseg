#!/bin/bash
#SBATCH --job-name=rangedit-test-new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-test-new-%j.out
#SBATCH --error=/scratch/p24cs0005/exp/job_log/rangediffseg/rangedit-test-new-%j.out

set -euo pipefail
cd /csehome/p24cs0005/rangediffseg

# Test-split inference (sequences 11-21) with the adapter + fusion checkpoint,
# writing per-point .label files in the SemanticKITTI benchmark format and
# packaging them exactly as the CodaBench submission expects:
#     sequences/<seq>/predictions/<frame>.label      at the zip root
#
# Unlike the earlier baseline submission this runs WITH KNN post-processing,
# which measured +3.35 3D mIoU on the val split for the same checkpoint.
CHECKPOINT=${CHECKPOINT:-/scratch/p24cs0005/exp/ckpt/rangediffseg/log_rangedit_scratch_adapter_rangeaug_cn07/checkpoint/best_mean_iou_model.pth}
DATA_ROOT=${DATA_ROOT:-/scratch/p24cs0005/kitti/dataset}
SAVE_PATH=${SAVE_PATH:-/scratch/p24cs0005/exp/ckpt/rangediffseg}
RUN_ID=${RUN_ID:-rangedit_test_adapter_rangeaug}
ZIP_OUT=${ZIP_OUT:-${SAVE_PATH}/${RUN_ID}_semantickitti_codabench.zip}

echo "job_id=${SLURM_JOB_ID:-none}  host=$(hostname)  start=$(date)"
echo "checkpoint=${CHECKPOINT}"
echo "zip_out=${ZIP_OUT}"

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u inference.py --config config.yaml \
  --data_root "${DATA_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --id "${RUN_ID}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --num_workers 8 \
  --save_eval_results

PRED_DIR="${SAVE_PATH}/log_${RUN_ID}/Eval_test/preds"
echo "packaging from ${PRED_DIR}"
test -d "${PRED_DIR}/sequences" || { echo "ERROR: predictions not found"; exit 1; }

rm -f "${ZIP_OUT}"
cd "${PRED_DIR}"
zip -r -q "${ZIP_OUT}" sequences/

echo "--- submission summary ---"
echo "labels written : $(find sequences -name '*.label' | wc -l)"
echo "sequences      : $(ls sequences | sort -n | tr '\n' ' ')"
echo "zip            : ${ZIP_OUT} ($(du -h "${ZIP_OUT}" | cut -f1))"
unzip -l "${ZIP_OUT}" | head -6
echo "end=$(date)"
