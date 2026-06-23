# PeriodicWaveBoson

Neural-network variational Monte Carlo for a two-dimensional bilayer boson
system with periodic boundary conditions.

The active simulation path is intentionally narrow:

- `periodicwave/configs/bilayer_bosons.py`: side-effect-free bilayer config builder.
- `scripts/train/run_bilayer.py`: canonical entry point for one bilayer run.
- `scripts/train/run_bilayer_rs_anneal.py`: density-continuation driver built on the
  same one-run entry point.
- `scripts/train/run_bilayer_rs_d_scan.py`: sharded `r_s,d` scan worker.
- `scripts/submit/`: HPC shell wrappers for single runs, anneals, and scans.
- `periodicwave/boson_network.py`: permutation-symmetric bosonic wavefunction.
- `periodicwave/pbc/bilayer_hamiltonian.py`: direct minimum-image dipolar
  bilayer Hamiltonian.
- `scripts/evaluate/evaluate_energies.py`: energy convergence plots from
  `train_stats.csv`.
- `scripts/evaluate/evaluate_observables.py`: density, layer density,
  z-density, pair-correlation, and structure-factor diagnostics.

The walker container stores xy positions and fixed layer labels for each boson.

Script directories:

- `scripts/train/`: Python entry points and workers that run training logic.
- `scripts/submit/`: shell wrappers for HPC environment setup and submission.
- `scripts/evaluate/`: per-run or per-scan diagnostics written into result folders.
- `scripts/visualize/`: scan-wide aggregate figures.
- `scripts/collect/`: tabular scan summaries.
- `scripts/benchmark/`: standalone benchmark utilities.

## Install

```bash
pip install -e .
```

Install a JAX build appropriate for your machine if it is not already present.

## Run One Simulation

From the repository root:

```bash
python3 scripts/train/run_bilayer.py
```

Common overrides:

```bash
python3 scripts/train/run_bilayer.py \
  --rs 0.5 \
  --d 1.5 \
  --seed 0 \
  --burn-in-iterations 2000 \
  --bold-iterations 2000 \
  --retune-iterations 1000 \
  --fine-iterations 3000 \
  --results-dir results
```

This writes to a folder like:

```text
results/N24_layers12_12_rs0.5_d1.5_D20.0_seed0_sq
```

Important files:

```text
config.json
device_info.log
train_stats.csv
qmcjax_ckpt_*.npz
qmcjax_best_*.npz
```

The old command `python3 periodicwave/configs/bilayer_bosons.py` is kept as a
compatibility shim, but new runs should use `scripts/train/run_bilayer.py`.

## Submit Batch Work

Use `scripts/submit/submit.sh` as the generic HPC launcher. It sets up the
runtime environment, logs the launch context, and then runs the Python script
you name. All remaining arguments are forwarded unchanged, so the command line
is the same as running `python3 scripts/train/...` directly.

Single run:

```bash
scripts/submit/submit.sh scripts/train/run_bilayer.py \
  --rs 0.5 \
  --d 1.5 \
  --seed 0 \
  --burn-in-iterations 2000 \
  --bold-iterations 2000 \
  --retune-iterations 1000 \
  --fine-iterations 3000 \
  --results-dir results
```

Density continuation:

```bash
scripts/submit/submit.sh scripts/train/run_bilayer_rs_anneal.py \
  --d 1.5 \
  --seed 0 \
  --rs-schedule 0.6,0.7,0.8,0.9,1.0 \
  --burn-in-iterations 2000 \
  --bold-iterations 2000 \
  --retune-iterations 1000 \
  --fine-iterations 3000 \
  --results-dir results/anneal_rs
```

Continue an anneal from an existing stage:

```bash
scripts/submit/submit.sh scripts/train/run_bilayer_rs_anneal.py \
  --restore-path results/anneal_rs/N32_layers16_16_rs0.5_d1.5_D20.0_seed0_sq \
  --rs-schedule 0.6,0.7,0.8,0.9,1.0 \
  --results-dir results/anneal_rs
```

Sharded `r_s,d` scan:

```bash
scripts/submit/submit.sh scripts/train/run_bilayer_rs_d_scan.py \
  --manifest scan_manifests/rs_d_manifest.csv \
  --rank 0 \
  --size 1 \
  --burn-in-iterations 2000 \
  --bold-iterations 2000 \
  --retune-iterations 1000 \
  --fine-iterations 3000
```

The scan worker only shards the manifest and launches `run_bilayer.py` once per
assigned row. Scan-specific flags are `--manifest`, `--rank`, `--size`,
`--run-script`, and `--dry-run`; all other flags are forwarded directly to
`run_bilayer.py`. Values from the manifest provide `--rs`, `--d`, and optional
per-row `--seed`.

On LLsub triplets, `scripts/train/run_bilayer_rs_d_scan.py` can also read
`LLSUB_RANK` and `LLSUB_SIZE` from the job environment, so `--rank` and
`--size` are optional when LLsub provides them.

All defaults and validation live in the Python entry points and
`periodicwave/configs/bilayer_bosons.py`. Use `--help` after the Python script
name to see that runner's options:

```bash
scripts/submit/submit.sh scripts/train/run_bilayer.py --help
scripts/submit/submit.sh scripts/train/run_bilayer_rs_anneal.py --help
scripts/submit/submit.sh scripts/train/run_bilayer_rs_d_scan.py --help
```

Stage lengths are independent. If a stage length is omitted, the runner uses the
config default from `periodicwave/configs/bilayer_bosons.py`; the post-burn
optimizer length is derived as the adiabatic warmup stages plus
`fine_iterations`. Resuming and extending a scan point follows the same
checkpoint behavior as a single `run_bilayer.py` run: rerun with a larger total
target length to continue beyond the latest checkpoint.

## Plot Energy

```bash
python3 scripts/evaluate/evaluate_energies.py \
  --scan-dir <results-parent-folder> \
  --pattern '<result-folder-name>'
```

This writes:

```text
fig_training_energy_trace.png
fig_training_diagnostics_overview.png
```

inside the run folder.

## Plot Bilayer Diagnostics

```bash
python3 scripts/evaluate/evaluate_observables.py \
  --scan-dir <results-parent-folder> \
  --pattern '<result-folder-name>'
```

This writes:

```text
fig_density_xy_overall.png
fig_density_xy_by_layer.png
fig_density_z_layers.png
fig_positions_xy_snapshots.png
fig_pair_correlation_gr.png
fig_structure_factor_sk.png
```

inside the run folder.

For the `scan_260603` scan, omit these arguments and the evaluators will use
their default scan directory and pattern.

LLsub scripts/submit/submit.sh [10,2,20] -g volta:1  -- scripts/train/run_bilayer_D_d_scan.py   --results-dir results/D_d_scan_N16_rs1.0_sq
LLsub scripts/submit/submit.sh [10,1,40] -g volta:2 -- scripts/train/run_bilayer_D_d_scan.py --results-dir results/D_d_scan_N24_rs1.0_sq_slice
