#!/usr/bin/env bash
#SBATCH --account=bhry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=wall-tail-crosscal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=wall-tail-crosscal_%j.out
set -euo pipefail
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bhry/crodriguez3/venv/bin/activate
cd /work/nvme/bhry/crodriguez3/wall-tail-selection/crossover-calibration-v2
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
date -Is
lscpu | head -16
sha256sum -c snapshot-sha256.txt >/dev/null
python -m unittest discover -s tests -v
python scripts/run_crossover.py --config configs/crossover-calibration.json --output "runs/deltaai-crosscal-${SLURM_JOB_ID}" --workers 8
python scripts/validate_crossover.py "runs/deltaai-crosscal-${SLURM_JOB_ID}"
date -Is
