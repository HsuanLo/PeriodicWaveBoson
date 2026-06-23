#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/home/gridsan/%u/project/PeriodicWaveBoson/jobs/jell-%j.out
#SBATCH --error=/home/gridsan/%u/project/PeriodicWaveBoson/jobs/jell-%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/submit/run_multinode_supercloud.sh <python-script> [args...]" >&2
  echo "Example: scripts/submit/run_multinode_supercloud.sh scripts/train/run_bilayer.py --batch-size 2400" >&2
  exit 2
fi

set +u
source /etc/profile
set -u
module load anaconda/Python-ML-2025a
source "$HOME/venv/periodicwaveboson/bin/activate"

cd "$HOME/project/PeriodicWaveBoson"
mkdir -p jobs

export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="$HOME/project/PeriodicWaveBoson:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
script="$1"
shift

PORT=$((29500 + SLURM_JOB_ID % 10000))
COORDINATOR="$(scontrol show hostname "$SLURM_NODELIST" | head -1):${PORT}"

echo "=== PeriodicWaveBoson multinode launch ==="
echo "date=$(date -Is)"
echo "coordinator=${COORDINATOR}"
echo "nodes=${SLURM_NNODES}"
echo "ntasks=${SLURM_NTASKS}"
echo "ntasks_per_node=${SLURM_NTASKS_PER_NODE}"
echo "python=${PYTHON_BIN}"
NUM_PROCESSES="${SLURM_NNODES}"
LOCAL_DEVICE_IDS="${LOCAL_DEVICE_IDS:-0,1}"
echo "jax_processes=${NUM_PROCESSES}"
echo "local_device_ids=${LOCAL_DEVICE_IDS}"
printf 'command='
printf '%q ' "${PYTHON_BIN}" "${script}" --coordinator_address "${COORDINATOR}" --num_processes "${NUM_PROCESSES}" --local_device_ids "${LOCAL_DEVICE_IDS}" --jobid "${SLURM_JOB_ID}" "$@"
printf '\n'

srun \
  --nodes="$SLURM_NNODES" \
  --ntasks="$NUM_PROCESSES" \
  --ntasks-per-node=1 \
  --gpus-per-task=2 \
  --distribution=block:block \
  "${PYTHON_BIN}" -u "${script}" \
    --coordinator_address "${COORDINATOR}" \
    --num_processes "${NUM_PROCESSES}" \
    --local_device_ids "${LOCAL_DEVICE_IDS}" \
    --jobid "${SLURM_JOB_ID}" \
    "$@"
