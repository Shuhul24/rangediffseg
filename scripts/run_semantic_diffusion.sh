#!/bin/bash
#SBATCH --job-name=semantic-dit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --exclude=cn01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/p24cs0005/exp/job_log/rangediffseg/semantic-dit-%j.out
set -euo pipefail
cd /csehome/p24cs0005/rangediffseg
module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh
source /csehome/p24cs0005/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
conda activate difforecast
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=${CONFIG:-config_semantic_diffusion.yaml}
RUN_DIR=${RUN_DIR:-/scratch/p24cs0005/exp/ckpt/rangediffseg/semantic_dit_${SLURM_JOB_ID}}
echo "commit=$(git rev-parse HEAD) job=${SLURM_JOB_ID} run=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv
ARGS=(--config "$CONFIG" --output "$RUN_DIR")
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "$RESUME"); fi
if [[ "${SMOKE:-0}" == 1 ]]; then
    python -m unittest discover -s tests -p 'test_semantic_diffusion.py' -v
    ARGS+=(--smoke)
fi
python -u train_semantic_diffusion.py "${ARGS[@]}"
