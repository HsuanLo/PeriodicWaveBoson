# PeriodicWaveBoson

Neural-network variational Monte Carlo for a two-dimensional bilayer boson
system with periodic boundary conditions.

The active simulation path is intentionally narrow:

- `periodicwave/configs/bilayer_bosons.py`: run configuration and entry point.
- `periodicwave/boson_network.py`: permutation-symmetric bosonic wavefunction.
- `periodicwave/pbc/bilayer_hamiltonian.py`: direct minimum-image dipolar
  bilayer Hamiltonian.
- `scripts/evaluate/evaluate_energies.py`: energy convergence plots from
  `train_stats.csv`.
- `scripts/evaluate/evaluate_observables.py`: density, layer density,
  z-density, pair-correlation, and structure-factor diagnostics.

The walker container stores xy positions and fixed layer labels for each boson.

## Install

```bash
pip install -e .
```

Install a JAX build appropriate for your machine if it is not already present.

## Run

From the repository root:

```bash
python3 periodicwave/configs/bilayer_bosons.py
```

The default run writes to a folder like:

```text
results/bilayer-bosons/BosonNet/N14_layers7_7_rs10.0_d1.0_D200.0_tri
```

Important files:

```text
config.json
device_info.log
train_stats.csv
qmcjax_ckpt_000999.npz
```

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
