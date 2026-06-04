# After-Matter: Bilayer Boson Changes And How To Use Them

This document explains the bilayer boson simulation path and how to run it.

The codebase is now focused on the JAX NN-VMC bilayer boson workflow: a
bosonic symmetric wavefunction plus a direct dipolar bilayer Hamiltonian.

## Table Of Contents

- [1. Main Components](#1-main-components)
- [1.1 New Bosonic Network](#11-new-bosonic-network)
- [1.2 New Bilayer Hamiltonian](#12-new-bilayer-hamiltonian)
- [1.3 New Run Script](#13-new-run-script)
- [1.4 New Bilayer Evaluation Script](#14-new-bilayer-evaluation-script)
- [1.5 Training Code Wiring](#15-training-code-wiring)
- [1.6 Base Config Defaults](#16-base-config-defaults)
- [2. How To Run The New Simulation](#2-how-to-run-the-new-simulation)
- [3. How To Plot The Bilayer Diagnostics](#3-how-to-plot-the-bilayer-diagnostics)
- [4. How To Change The Physical Parameters](#4-how-to-change-the-physical-parameters)
- [5. How To Change The Network](#5-how-to-change-the-network)
- [6. How Adaptive MCMC Works Now](#6-how-adaptive-mcmc-works-now)
- [7. What Was Not Implemented Yet](#7-what-was-not-implemented-yet)
- [8. Suggested First Workflow](#8-suggested-first-workflow)
- [9. Validation Checks To Run](#9-validation-checks-to-run)
- [10. Files Touched](#10-files-touched)

## 1. Main Components

### 1.1 New Bosonic Network

Added:

```text
periodicwave/boson_network.py
```

This file implements a permutation-symmetric bosonic neural wavefunction.

The current network returns:

$$
\psi(R) = \exp(\log|\psi(R)|)
$$

with a fixed real positive phase. In code, `BosonNet.apply(...)` returns:

```python
phase, log_abs
```

where:

```python
phase = 1.0
```

and `log_abs` is the scalar neural-network output.

Available architectures:

- `"DeepSets"`: simple permutation-equivariant MLP blocks followed by symmetric
  pooling.
- `"Transformer"`: pre-norm self-attention blocks followed by symmetric pooling.
- `"Attention"`: accepted as an alias for `"Transformer"`.

The current run config uses `"Transformer"` for a more expressive correlated
bosonic ansatz. Permutation invariance is preserved by symmetric pooling.

### 1.2 New Bilayer Hamiltonian

Added:

```text
periodicwave/pbc/bilayer_hamiltonian.py
```

This implements a bilayer boson Hamiltonian with:

- continuous coordinates only in the $x$-$y$ plane,
- periodic boundary conditions in $x$-$y$,
- fixed discrete layer labels corresponding to $z=\pm d/2$,
- kinetic energy over $x,y$ only,
- direct minimum-image dipole-dipole interaction.

The kinetic term is:

$$
T =
-\frac{1}{2}\sum_i
\left(
\frac{\partial^2}{\partial x_i^2}
+ \frac{\partial^2}{\partial y_i^2}
\right).
$$

The dipolar interaction is:

$$
V_{\mathrm{dd}}(\mathbf r)
= C_{\mathrm{dd}}
\frac{1-3(z/r)^2}{r^3}.
$$

The default run uses the direct minimum-image interaction:

```python
use_ewald = False
```

An optional 2D-periodic/open-z Ewald path is available through:

```python
use_ewald = True
ewald_geometry = "xy_periodic_open_z"
ewald_alpha = None
ewald_real_cut = 4
ewald_kmax = 8
```

This Ewald path is for z-polarized dipoles in a geometry periodic in `x,y` and
open in `z`. The default direct path is unchanged unless `use_ewald=True`.

### 1.3 New Run Script

Added:

```text
periodicwave/configs/bilayer_bosons.py
```

This is the first runnable bilayer boson example.

Default model:

```python
num_bosons = 14
layer_occupations = (7, 7)
layer_separation = 1.0
dipole_strength = 200.0
supercell_shape = "tri"
density_rs = 10.0
```

The default run is intentionally small:

```python
cfg.batch_size = 1024
cfg.optim.iterations = 1000
cfg.optim.optimizer = "adam"
```

Adjust `cfg.optim.iterations`, `cfg.batch_size`, and the physical parameters
for production calculations.

### 1.4 New Bilayer Evaluation Script

Added:

```text
scripts/evaluate/evaluate_observables.py
```

This loads bilayer boson checkpoints and saves:

```text
fig_density_xy_overall.png
fig_density_xy_by_layer.png
fig_density_z_layers.png
fig_structure_factor_sk.png
```

The structure factor is computed as:

$$
S(\mathbf k)
=
\frac{1}{N}
\left\langle
\left|
\sum_j e^{i\mathbf k\cdot\mathbf r_j}
\right|^2
\right\rangle_{\mathrm{MC}}.
$$

### 1.5 Training Code Wiring

Modified:

```text
periodicwave/train.py
```

Changes:

- imports `BosonNet`,
- requires `cfg.network.network_type = "BosonNet"`,
- passes the bilayer layer separation into `BosonNet`,
- allows `train.train(..., layer_assignment=...)`,
- stores bilayer layer labels in `data.spins`,
- freezes adaptive MCMC move-width updates after `cfg.mcmc.adaptive_steps`.

Important naming note:

The code uses `spins` as the storage field for fixed bilayer labels. In this
boson-only code path, `spins` does not mean fermionic spin.

### 1.6 Base Config Defaults

Modified:

```text
periodicwave/default_config.py
```

Added:

```python
cfg.network.BosonNet
cfg.mcmc.adaptive_steps
```

The old electron network defaults have been removed.

## 2. How To Run The New Simulation

From the repository root:

```bash
python3 periodicwave/configs/bilayer_bosons.py
```

The default output folder is:

```text
results/bilayer-bosons/BosonNet/N14_layers7_7_rs10.0_d1.0_D200.0_tri
```

Expected files after a successful run:

```text
config.json
device_info.log
train_stats.csv
qmcjax_ckpt_000019.npz
```

The exact checkpoint number depends on `cfg.optim.iterations`.

## 3. How To Plot The Bilayer Diagnostics

After a checkpoint exists, run:

```bash
python3 scripts/evaluate/evaluate_observables.py \
  --scan-dir <results-parent-folder> \
  --pattern '<result-folder-name>'
```

It writes images into the same result folder:

```text
results/bilayer-bosons/BosonNet/N14_layers7_7_rs10.0_d1.0_D200.0_tri/fig_density_xy_overall.png
results/bilayer-bosons/BosonNet/N14_layers7_7_rs10.0_d1.0_D200.0_tri/fig_density_xy_by_layer.png
results/bilayer-bosons/BosonNet/N14_layers7_7_rs10.0_d1.0_D200.0_tri/fig_density_z_layers.png
results/bilayer-bosons/BosonNet/N14_layers7_7_rs10.0_d1.0_D200.0_tri/fig_structure_factor_sk.png
```

If you change parameters in `bilayer_bosons.py`, pass the matching parent
folder with `--scan-dir` and the run folder name with `--pattern`.

## 4. How To Change The Physical Parameters

Edit:

```text
periodicwave/configs/bilayer_bosons.py
```

Common parameters:

```python
num_bosons = 14
layer_occupations = (7, 7)
layer_separation = 1.0
dipole_strength = 200.0
supercell_shape = "tri"
density_rs = 10.0
```

Layer labels are fixed by:

```python
layer_assignment = np.array(
    [1.0] * layer_occupations[0] + [-1.0] * layer_occupations[1])
```

This means the first `layer_occupations[0]` bosons are in the top layer and the
remaining bosons are in the bottom layer.

The simulation currently does not perform layer-flip or tunneling moves.

## 5. How To Change The Network

In `bilayer_bosons.py`, the default is:

```python
cfg.network.network_type = "BosonNet"
cfg.network.BosonNet.architecture = "Transformer"
cfg.network.BosonNet.num_layers = 4
cfg.network.BosonNet.mlp_dim = 128
```

To try the cheaper DeepSets model:

```python
cfg.network.BosonNet.architecture = "DeepSets"
cfg.network.BosonNet.num_layers = 3
cfg.network.BosonNet.mlp_dim = 64
```

The Transformer path is more expensive because attention scales roughly like
$N^2$ in particle number.

## 6. How Adaptive MCMC Works Now

The old code adapted the MCMC move width throughout training. The new behavior
supports an adaptive phase followed by a frozen phase.

Relevant config:

```python
cfg.mcmc.move_width_updater = "adaptive"
cfg.mcmc.adapt_frequency = 10
cfg.mcmc.adaptive_steps = 100
```

This means:

- every 10 optimization steps, the code checks `pmove`,
- if `pmove > 0.55`, it increases the move width,
- if `pmove < 0.50`, it decreases the move width,
- after step 100, it stops changing the move width.

This matches the requested adaptive-then-fixed sampling behavior.

## 7. What Was Not Implemented Yet

The following are intentionally left for later stages:

1. 2D-periodic anisotropic dipolar Ewald summation.
2. Layer-flip or layer-swap Monte Carlo moves.
3. Variable layer occupations.
4. Interlayer tunneling.
5. Complex bosonic wavefunctions.
6. Short-range physical pseudopotentials beyond the direct dipolar model.

The current implementation is a working minimum-effort base for testing the
bosonic NN-VMC pipeline before adding the harder Ewald physics.

## 8. Suggested First Workflow

1. Run the default small example:

   ```bash
   python3 periodicwave/configs/bilayer_bosons.py
   ```

2. Check that a result folder appears under:

   ```text
   results/bilayer-bosons/BosonNet/
   ```

3. Inspect:

   ```text
   train_stats.csv
   ```

4. Generate plots:

   ```bash
   python3 scripts/evaluate/evaluate_observables.py \
     --scan-dir <results-parent-folder> \
     --pattern '<result-folder-name>'
   ```

5. Increase only one thing at a time:

   - `cfg.optim.iterations`,
   - `cfg.batch_size`,
   - `num_bosons`,
   - `cfg.network.BosonNet.mlp_dim`,
   - `cfg.network.BosonNet.architecture`.

6. Only after the direct interaction path is stable, implement and validate the
   dipolar Ewald summation.

## 9. Validation Checks To Run

Before trusting physics results, validate:

- energy is finite,
- `pmove` stays between 0 and 1,
- local standard deviation is finite,
- checkpoints load,
- density plots are not empty,
- `BosonNet` output is invariant under particle permutations,
- dipolar signs are correct:

  $$
  V_{\mathrm{dd}}(\Delta z=0) > 0,
  \qquad
  V_{\mathrm{dd}}(\rho=0,\Delta z=d) < 0.
  $$

## 10. Files Touched

Added:

```text
periodicwave/boson_network.py
periodicwave/pbc/bilayer_hamiltonian.py
periodicwave/configs/bilayer_bosons.py
scripts/evaluate/evaluate_observables.py
docs/after-matter.md
```

Modified:

```text
periodicwave/train.py
periodicwave/default_config.py
```

Legacy electron-gas scripts, fermionic network modules, and old planning docs
were removed so the repository now presents only the bilayer boson workflow.
