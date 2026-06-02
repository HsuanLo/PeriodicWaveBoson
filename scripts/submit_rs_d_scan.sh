#!/bin/bash

source /etc/profile
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate py313 2>/dev/null || true
source /home/gridsan/hlo1/venv/periodicwaveboson/bin/activate

cd /home/gridsan/hlo1/project/PeriodicWaveBoson

export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp
export PYTHONPATH="/home/gridsan/hlo1/project/PeriodicWaveBoson:${PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "LLSUB_RANK=${LLSUB_RANK:-0}"
echo "LLSUB_SIZE=${LLSUB_SIZE:-1}"
echo "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/run_scan_worker.py \
  --manifest scans/rs_d_manifest.csv \
  --rank "${LLSUB_RANK:-0}" \
  --size "${LLSUB_SIZE:-1}" \
  --min-completed-step "${SCAN_MIN_COMPLETED_STEP:-4999}" \
  "$@"
