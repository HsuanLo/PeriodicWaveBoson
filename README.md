# PeriodicWaveBoson

Neural-network variational Monte Carlo for a two-dimensional bilayer boson
system with periodic boundary conditions.

The active simulation path is intentionally narrow:

- `periodicwave/configs/run_bilayer_bosons.py`: run configuration and entry point.
- `periodicwave/BosonNet.py`: permutation-symmetric bosonic wavefunction.
- `periodicwave/pbc/bilayer_hamiltonian.py`: direct minimum-image dipolar
  bilayer Hamiltonian.
- `evaluate-energies.py`: energy convergence plot from `train_stats.csv`.
- `evaluate-bilayer.py`: density, layer density, z-density, and structure-factor
  diagnostics.

The walker container stores xy positions and fixed layer labels for each boson.

## Install

```bash
pip install -e .
```

Install a JAX build appropriate for your machine if it is not already present.

## Run

From the repository root:

```bash
python3 periodicwave/configs/run_bilayer_bosons.py
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
python3 evaluate-energies.py
```

This writes:

```text
energy.png
```

inside the run folder.

## Plot Bilayer Diagnostics

```bash
python3 evaluate-bilayer.py
```

This writes:

```text
bilayer_density_xy.png
bilayer_density_layers.png
bilayer_density_z.png
bilayer_structure_factor.png
```

inside the run folder.

If you change the run parameters in `run_bilayer_bosons.py`, update the matching
parameters at the top of `evaluate-energies.py` and `evaluate-bilayer.py`.
