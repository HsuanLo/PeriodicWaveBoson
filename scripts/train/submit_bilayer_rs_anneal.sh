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

ANNEAL_D="${ANNEAL_D:-4.0}"
ANNEAL_SEED="${ANNEAL_SEED:-42}"
ANNEAL_ITERATIONS_PER_STAGE="${ANNEAL_ITERATIONS_PER_STAGE:-5000}"
ANNEAL_RS_SCHEDULE="${ANNEAL_RS_SCHEDULE:-1.0,0.9,0.8,0.7,0.6,0.5}"
ANNEAL_RESULTS_DIR="${ANNEAL_RESULTS_DIR:-results/anneal_rs}"

echo "LLSUB_RANK=${LLSUB_RANK:-0}"
echo "LLSUB_SIZE=${LLSUB_SIZE:-1}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ANNEAL_D=${ANNEAL_D}"
echo "ANNEAL_SEED=${ANNEAL_SEED}"
echo "ANNEAL_ITERATIONS_PER_STAGE=${ANNEAL_ITERATIONS_PER_STAGE}"
echo "ANNEAL_RS_SCHEDULE=${ANNEAL_RS_SCHEDULE}"
echo "ANNEAL_RESULTS_DIR=${ANNEAL_RESULTS_DIR}"

"${PYTHON_BIN}" scripts/train/run_bilayer_rs_anneal.py \
  --d "${ANNEAL_D}" \
  --seed "${ANNEAL_SEED}" \
  --iterations-per-stage "${ANNEAL_ITERATIONS_PER_STAGE}" \
  --rs-schedule "${ANNEAL_RS_SCHEDULE}" \
  --results-dir "${ANNEAL_RESULTS_DIR}" \
  "$@"
