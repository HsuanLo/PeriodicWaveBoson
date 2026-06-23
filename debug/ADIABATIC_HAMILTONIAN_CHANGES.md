# Adiabatic Hamiltonian Debug Changes

This file records the debug-only changes for testing adiabatic Hamiltonian
warmup before the normal bilayer NN-VMC training stages.

## Goal

Test whether a short continuation warmup helps when intra-layer and inter-layer
potential energy scales create a stiff optimization landscape. The final
production stages still train on the physical Hamiltonian.

## Training Flow

When adiabatic mode is disabled, the existing flow is unchanged:

```text
burn_in -> retune -> fine
```

When adiabatic mode is enabled, short KFAC warmup stages are inserted before
the physical stages:

```text
burn_in
-> adiabatic_00_inter_<scale>
-> adiabatic_01_inter_<scale>
-> ...
-> retune
-> fine
```

The normal `retune` and `fine` stages are kept as the physical post-warmup run.

## Hamiltonian

The debug bilayer Hamiltonian now supports:

```text
H = T + alpha_intra V_intra + alpha_inter V_inter
```

The defaults are physical:

```text
alpha_intra = 1.0
alpha_inter = 1.0
```

Adiabatic warmup stages set these through stage-local
`make_local_energy_kwargs`.
By default, the first adiabatic stage chooses both starting scales from the
initial walker energy decomposition:

```text
alpha_intra_start = clamp(0.2 * median(|T|) / median(|V_intra|), 1e-2, 1)
alpha_inter_start = clamp(0.2 * median(|T|) / median(|V_inter|), 1e-4, 1)
```

Both scales then ramp geometrically to `1.0`.

## Files Changed

- `debug/periodicwave/pbc/bilayer_hamiltonian.py`
  - Added `potential_intra_scale` and `potential_inter_scale` arguments to
    `local_energy`.
  - Scales the already-separated `potential_parts["intra"]` and
    `potential_parts["inter"]` before forming the local energy.

- `debug/periodicwave/train.py`
  - Allows each training stage to override `cfg.system.make_local_energy_kwargs`
    with `stage["make_local_energy_kwargs"]`.
  - Logs `potential_intra_scale` and `potential_inter_scale` in `train_stats.csv`.
  - Includes the active scales in the stage-start log message.

- `debug/periodicwave/configs/bilayer_bosons.py`
  - Added config defaults for adiabatic warmup.
  - Added adiabatic KFAC warmup stage generation.
  - Keeps `retune` and `fine` after the adiabatic warmup.
  - Adds `_adiabatic` to result folder names when enabled.

- `debug/scripts/train/run_bilayer.py`
  - Added CLI/env controls for adiabatic warmup.

- `debug/scripts/evaluate/kinetic_cm_rel.py`
  - Added a checkpoint postprocessor that decomposes kinetic energy into
    fixed-pair center-of-mass and relative-coordinate pieces.
  - Reports an independent Cartesian kinetic recomputation as a consistency
    check.

- `debug/scripts/evaluate/pair_center_structure.py`
  - Added a checkpoint postprocessor for top-bottom pair-center static
    structure factors.
  - Saves `fig_pair_center_structure_factor*.png` and
    `pair_center_structure_factor*.csv`.

## Controls

CLI flags:

```text
--adiabatic
--adiabatic-inter-start auto
--adiabatic-intra-start auto
--adiabatic-num-stages 20
--adiabatic-schedule log
--adiabatic-intra-final 1.0
--adiabatic-inter-final 1.0
--adiabatic-iterations 500
--adiabatic-lr-rate 0.003
--adiabatic-auto-scale-factor 0.2
--adiabatic-intra-min-scale 1e-2
--adiabatic-inter-min-scale 1e-4
```

The intra-layer and inter-layer schedules ramp from their automatic starts to
their configured final scales.

Environment variables:

```text
BILAYER_ADIABATIC
BILAYER_ADIABATIC_INTER_START
BILAYER_ADIABATIC_INTRA_START
BILAYER_ADIABATIC_NUM_STAGES
BILAYER_ADIABATIC_SCHEDULE
BILAYER_ADIABATIC_INTRA_FINAL
BILAYER_ADIABATIC_INTER_FINAL
BILAYER_ADIABATIC_ITERATIONS
BILAYER_ADIABATIC_LR_RATE
BILAYER_ADIABATIC_AUTO_SCALE_FACTOR
BILAYER_ADIABATIC_INTRA_MIN_SCALE
BILAYER_ADIABATIC_INTER_MIN_SCALE
```

## Default Adiabatic Schedule

If `--adiabatic` is passed, the default warmup scales are selected from the
initial walker energy decomposition and then ramped to the physical
Hamiltonian:

```text
alpha_intra = geomspace(auto_intra_start, 1.0, num_stages)
alpha_inter = geomspace(auto_inter_start, 1.0, num_stages)
```

After those warmup stages, the existing `retune` and `fine` stages use the
physical Hamiltonian with both scale factors equal to `1.0`.

## KFAC Handling

Each adiabatic warmup stage sets:

```text
reset_optimizer_state = True
```

This keeps the model parameters and walkers, but reinitializes the KFAC state
when the Hamiltonian changes.

## Example

```bash
python debug/scripts/train/run_bilayer.py \
  --adiabatic \
  --adiabatic-inter-start 1e-4 \
  --adiabatic-num-stages 20 \
  --adiabatic-schedule log \
  --adiabatic-iterations 500
```

Compare against the direct run with the same physical parameters and without
`--adiabatic`.

## Kinetic COM/Relative Diagnostic

To test whether similar total kinetic energy is hiding different pair
center-of-mass and relative-coordinate contributions, run:

```bash
python debug/scripts/evaluate/kinetic_cm_rel.py \
  --run-dir debug/results/N12_layers6_6_rs1.5_d0.5_D6.0_seed42_sq_adiabatic \
  --checkpoint qmcjax_ckpt_007499.npz \
  --max-walkers 128 \
  --batch-size 16
```

The script uses fixed-index pairing: the kth top-layer particle is paired with
the kth bottom-layer particle. It reports:

```text
T_cm/N
T_rel/N
(T_cm + T_rel)/N
T_cartesian/N
T_cartesian - (T_cm + T_rel)
```

The Cartesian difference should be near zero. If it is not, the decomposition
or coordinate convention needs attention before interpreting the result.

The evaluator shows a `tqdm` progress bar over walker batches. Use
`--no-progress` to disable it, or `--batch-size <=0` to evaluate all selected
walkers in one batch.

## Pair-Center Structure Factor

To diagnose whether the bound top-bottom pairs form a liquid or crystal, run:

```bash
python debug/scripts/evaluate/pair_center_structure.py \
  --run-dir debug/results/N12_layers6_6_rs1.5_d0.5_D6.0_seed42_sq_adiabatic \
  --checkpoint qmcjax_ckpt_007499.npz \
  --kmax 5
```

The script forms fixed-index pair centers using a PBC-aware relative vector:

```text
R_pair_i = r_bottom_i + 0.5 * minimum_image(r_top_i - r_bottom_i)
```

and computes:

```text
S_pair(k) = < |sum_i exp(i k . R_pair_i)|^2 > / N_pair
```

Large sharp peaks, especially peaks with `S_pair(k_peak) / N_pair` that remain
large or grow with system size, indicate pair-crystal order.
