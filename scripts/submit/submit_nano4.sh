#!/bin/bash
#SBATCH --account=MST114233
#SBATCH --job-name=vasp660_h200_1gpu
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/submit/submit.sh <python-script> [args...]" >&2
  echo "Example: scripts/submit/submit.sh scripts/train/run_bilayer.py --rs 0.5" >&2
  exit 2
fi
source "/work/$USER/miniconda3/etc/profile.d/conda.sh"
conda activate py313

cd /work/$USER/PeriodicWaveBoson

export PYTHONPATH="/work/$USER/PeriodicWaveBoson:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

script="$1"
shift

echo "=== PeriodicWaveBoson submit launch ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=${PYTHON_BIN}"
echo "slurm_job_id=${SLURM_JOB_ID:-<unset>}"
echo "slurm_job_name=${SLURM_JOB_NAME:-<unset>}"
echo "slurm_nodelist=${SLURM_NODELIST:-<unset>}"
echo "slurm_procid=${SLURM_PROCID:-<unset>}"
echo "slurm_localid=${SLURM_LOCALID:-<unset>}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "xla_preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE:-<unset>}"
echo "xla_mem_fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION:-<unset>}"
printf 'command='
printf '%q ' "${PYTHON_BIN}" "${script}" "$@"
printf '\n'

exec "${PYTHON_BIN}" "${script}" "$@"
