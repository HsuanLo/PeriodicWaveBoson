#!/bin/bash

set -euo pipefail

cd /home/gridsan/hlo1/project/PeriodicWaveBoson

export ANNEAL_D="${ANNEAL_D:-0.5}"
export ANNEAL_SEED="${ANNEAL_SEED:-12}"
export ANNEAL_ITERATIONS_PER_STAGE="${ANNEAL_ITERATIONS_PER_STAGE:-5000}"
export ANNEAL_RESULTS_DIR="${ANNEAL_RESULTS_DIR:-results/anneal_rs_seed${ANNEAL_SEED}}"
export ANNEAL_RS_SCHEDULE="${ANNEAL_RS_SCHEDULE:-0.4,0.3,0.2,0.1}"

export ANNEAL_RESTORE_RS="${ANNEAL_RESTORE_RS:-0.5}"
export RESTORE_PATH="${RESTORE_PATH:-${ANNEAL_RESULTS_DIR}/N24_layers12_12_rs${ANNEAL_RESTORE_RS}_d${ANNEAL_D}_D20.0_seed${ANNEAL_SEED}_sq}"

if ! compgen -G "${RESTORE_PATH}/qmcjax_ckpt_*.npz" > /dev/null; then
  echo "Missing restart checkpoint matching: ${RESTORE_PATH}/qmcjax_ckpt_*.npz" >&2
  exit 1
fi

exec bash scripts/train/submit_bilayer_rs_anneal.sh "$@"
