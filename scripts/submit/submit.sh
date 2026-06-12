#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/submit/submit.sh <python-script> [args...]" >&2
  echo "Example: scripts/submit/submit.sh scripts/train/run_bilayer.py --rs 0.5" >&2
  exit 2
fi

source /etc/profile
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate py313 2>/dev/null || true
source /home/gridsan/hlo1/venv/periodicwaveboson/bin/activate

cd /home/gridsan/hlo1/project/PeriodicWaveBoson

export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp
export PYTHONPATH="/home/gridsan/hlo1/project/PeriodicWaveBoson:${PYTHONPATH:-}"
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
