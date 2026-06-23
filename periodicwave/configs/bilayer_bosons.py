# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Configuration builder for bilayer boson NN-VMC calculations with xy PBC."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

import ml_collections
import numpy as np

from periodicwave import default_config
from periodicwave.pbc import lattices


LOCAL_ENERGY_FN = "periodicwave.pbc.bilayer_hamiltonian.local_energy"


@dataclasses.dataclass(frozen=True)
class BilayerDefaults:
  """Shared defaults for bilayer runners and config construction."""

  # Physical system.
  num_bosons: int = 32
  layer_occupations: tuple[int, int] = (16, 16)
  layer_separation: float = 1.0
  dipole_strength: float = 6.0
  supercell_shape: str = "sq"
  density_rs: float = 1.0
  seed: int = 42

  # Run control.
  burn_in_iterations: int = 0
  fine_iterations: int = 10000
  fine_lr_rate: float = 1.0e-3
  results_dir: str = "results"
  restore_path: str = ""
  reset_iteration_on_restore: bool = False
  reset_optimizer_on_restore: bool = False
  adiabatic_inter_start: float | None = None
  adiabatic_intra_start: float | None = None
  adiabatic_num_stages: int = 20
  adiabatic_schedule: str = "log"
  adiabatic_intra_final: float = 1.0
  adiabatic_inter_final: float = 1.0
  adiabatic_iterations: int = 500
  adiabatic_stage_lr_rate: float = 3.0e-3
  adiabatic_start_target: float = 0.05

  # Logging/checkpointing.
  batch_size: int = 2048
  stats_frequency: int = 10
  save_frequency_minutes: float = 30.0
  best_checkpoint_min_step: int = 500
  best_checkpoint_metric: str = "ewmean_std"
  best_checkpoint_std_weight: float = 1.0

  # Transformer BosonNet.
  network_layers: int = 4
  mlp_dim: int = 32
  num_heads: int = 4
  attn_dim: int = 16
  value_dim: int = 16
  num_perceptrons_per_layer: int = 2
  use_distance_attention_bias: bool = True
  distance_attention_bias_num_rbf: int = 16
  distance_attention_bias_eps: float = 1.0e-6
  distance_attention_bias_scale: float = 1.0
  use_dipole_attention_bias: bool = True
  dipole_attention_bias_scale: float = 1.0

  # MCMC defaults shared by stages unless overridden.
  init_layout: str = "farthest"
  init_width_fraction: float = 1.0e-3
  move_width: float = 0.05
  global_move_fraction: float = 0.05
  global_width_scale: float = 0.25
  adapt_frequency: int = 1
  adapt_rate: float = 0.05
  min_move_width: float = 1.0e-3


DEFAULTS = BilayerDefaults()


@dataclasses.dataclass(frozen=True)
class AdiabaticWarmup:
  """Parameters for the adiabatic Hamiltonian continuation stages."""

  inter_start: float
  intra_start: float
  num_stages: int
  schedule: str
  intra_final: float
  inter_final: float
  iterations: int
  lr_rate: float


def format_layer_occupations(layer_occupations: Sequence[int]) -> str:
  """Returns the canonical folder/CLI spelling for layer occupations."""
  layers = _as_layer_occupations(layer_occupations)
  return f"{layers[0]}_{layers[1]}"


def _as_layer_occupations(value: Sequence[int]) -> tuple[int, int]:
  if len(value) != 2:
    raise ValueError("layer_occupations must contain two entries.")
  return int(value[0]), int(value[1])


def _make_layer_assignment(layer_occupations: tuple[int, int]) -> np.ndarray:
  top, bottom = layer_occupations
  return np.array([1.0] * top + [-1.0] * bottom)


def _make_lattice(
    *,
    num_bosons: int,
    density_rs: float,
    supercell_shape: str,
) -> np.ndarray:
  if supercell_shape == "sq":
    supercell_a = density_rs * np.sqrt(np.pi * num_bosons)
    return lattices._square_lattice_vecs(supercell_a)

  if supercell_shape == "tri":
    supercell_a = density_rs * np.sqrt(
        2 * np.pi / np.sqrt(3) * num_bosons)
    supercell_lattice, _ = lattices._triangular_lattice_vecs_periodic_potential(
        supercell_a, 1)
    return supercell_lattice

  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")


def _physics_adiabatic_start_scales(
    *,
    density_rs: float,
    layer_separation: float,
    dipole_strength: float,
    target: float,
) -> tuple[float, float]:
  """Returns explicit warmup starts from simple kinetic/potential estimates."""
  rs = float(density_rs)
  d = float(layer_separation)
  dipole = float(dipole_strength)
  if rs <= 0.0:
    raise ValueError("density_rs must be positive.")
  if d <= 0.0:
    raise ValueError("layer_separation must be positive.")
  if dipole <= 0.0:
    raise ValueError("dipole_strength must be positive.")
  if target <= 0.0:
    raise ValueError("adiabatic_start_target must be positive.")

  separation_ratio = d / rs
  kinetic_est = 1.7 / (rs ** 2) * max(1.0, separation_ratio ** -2.5)
  intra_est = 2.05 * (dipole / 6.0) / (rs ** 3)
  if separation_ratio < 1.0:
    inter_shape = separation_ratio ** -4
  else:
    inter_shape = separation_ratio ** -6
  inter_est = intra_est * inter_shape

  def scale(component_est: float) -> float:
    value = target * kinetic_est / component_est
    return float(np.clip(value, 1.0e-4, 1.0))

  return scale(intra_est), scale(inter_est)


def _burn_in_stage(iterations: int) -> dict:
  return {
      "name": "burn_in",
      "iterations": iterations,
      "optimizer": "none",
      "mcmc_steps": 10,
      "proposal": "block",
      "block_size": 1,
      "target_acceptance": 0.30,
      "adapt_width": False,
  }


def _fine_stage(*, iterations: int, lr_rate: float) -> dict:
  return {
      "name": "fine",
      "iterations": iterations,
      "optimizer": "kfac",
      "lr_rate": lr_rate,
      "mcmc_steps": 30,
      "proposal": "hybrid",
      "block_size": 2,
      "global_move_fraction": 0.05,
      "global_width_scale": 0.05,
      "target_acceptance": 0.25,
      "adapt_width": False,
  }


def _make_training_stages(
    *,
    burn_in_iterations: int,
    fine_iterations: int,
    fine_lr_rate: float,
    warmup: AdiabaticWarmup,
) -> list[dict]:
  """Returns the staged sampler/optimizer protocol."""
  stages = [_burn_in_stage(burn_in_iterations)]
  inter_scales = _adiabatic_inter_scales(
      start=warmup.inter_start,
      final=warmup.inter_final,
      num_stages=warmup.num_stages,
      schedule=warmup.schedule)
  stages.extend(
      _make_adiabatic_stages(
          inter_scales=inter_scales,
          warmup=warmup))
  stages.append(_fine_stage(iterations=fine_iterations, lr_rate=fine_lr_rate))
  return stages


def _adiabatic_inter_scales(
    *,
    start: float,
    final: float,
    num_stages: int,
    schedule: str,
) -> tuple[float | None, ...]:
  """Returns inter-layer potential scales for adiabatic warmup."""
  if num_stages <= 0:
    raise ValueError("adiabatic_num_stages must be positive.")
  schedule = schedule.lower()
  start = float(start)
  if schedule == "log":
    if start <= 0.0:
      raise ValueError("adiabatic_inter_start must be positive for log schedule.")
    return tuple(float(scale) for scale in np.geomspace(start, final, num_stages))
  if schedule == "linear":
    return tuple(float(scale) for scale in np.linspace(start, final, num_stages))
  raise ValueError(f"Unknown adiabatic_schedule: {schedule}")


def _make_adiabatic_stages(
    *,
    inter_scales: Sequence[float | None],
    warmup: AdiabaticWarmup,
) -> list[dict]:
  """Returns KFAC warmup stages for adiabatic Hamiltonian continuation."""
  stages = []
  for index, inter_scale in enumerate(inter_scales):
    inter_name = f"{inter_scale:g}"
    stages.append({
        "name": f"adiabatic_{index:02d}_inter_{inter_name}",
        "iterations": warmup.iterations,
        "optimizer": "kfac",
        "lr_rate": warmup.lr_rate,
        "mcmc_steps": 10,
        "proposal": "hybrid",
        "block_size": 4,
        "global_move_fraction": 0.1,
        "global_width_scale": 0.10,
        "target_acceptance": 0.25,
        "adapt_width": True,
        "adaptive_steps": warmup.iterations,
        "reset_optimizer_state": True,
        "make_local_energy_kwargs": {
            "potential_intra_scale": None,
            "potential_inter_scale": inter_scale,
        },
        "adiabatic_scale_schedule": {
            "index": index,
            "num_stages": len(inter_scales),
            "schedule": warmup.schedule,
            "intra_start": warmup.intra_start,
            "intra_final": warmup.intra_final,
            "inter_start": warmup.inter_start,
            "inter_final": warmup.inter_final,
        },
    })
  return stages


def _result_folder(
    *,
    results_dir: str | Path,
    num_bosons: int,
    layer_occupations: tuple[int, int],
    density_rs: float,
    layer_separation: float,
    dipole_strength: float,
    seed: int,
    supercell_shape: str,
) -> str:
  return str(
      Path(results_dir)
      / (
          f"N{num_bosons}_layers{layer_occupations[0]}_{layer_occupations[1]}"
          f"_rs{density_rs}_d{layer_separation}_D{dipole_strength}"
          f"_seed{seed}_{supercell_shape}_adiabatic"
      ))


def _configure_system(
    cfg: ml_collections.ConfigDict,
    *,
    num_bosons: int,
    layer_separation: float,
    dipole_strength: float,
    lattice: np.ndarray,
) -> None:
  cell_area = abs(np.linalg.det(lattice))

  cfg.system.bosons = (num_bosons, 0)
  cfg.system.ndim = 2
  cfg.system.pbc_lattice = lattice
  cfg.system.make_local_energy_fn = LOCAL_ENERGY_FN
  cfg.system.make_local_energy_kwargs = {
      "lattice": lattice,
      "layer_separation": layer_separation,
      "potential_type": "Dipolar",
      "potential_kwargs": {
          "dipole_strength": dipole_strength,
          "softening": 1.0e-2,
          "use_ewald": True,
          "ewald_alpha": 10.0 / np.sqrt(cell_area),
          "ewald_real_cut": 2,
          "ewald_kmax": 12,
          "ewald_geometry": "xy_periodic_open_z",
      },
      "kinetic_kwargs": {"laplacian_method": "folx"},
  }


def _configure_network(
    cfg: ml_collections.ConfigDict,
    *,
    dipole_strength: float,
) -> None:
  cfg.network.network_type = "BosonNet"
  cfg.network.complex = False
  cfg.network.BosonNet.architecture = "Transformer"
  cfg.network.BosonNet.num_layers = DEFAULTS.network_layers
  cfg.network.BosonNet.mlp_dim = DEFAULTS.mlp_dim
  cfg.network.BosonNet.num_heads = DEFAULTS.num_heads
  cfg.network.BosonNet.attn_dim = DEFAULTS.attn_dim
  cfg.network.BosonNet.value_dim = DEFAULTS.value_dim
  cfg.network.BosonNet.num_perceptrons_per_layer = (
      DEFAULTS.num_perceptrons_per_layer)
  cfg.network.BosonNet.use_layer_norm = True
  cfg.network.BosonNet.mlp_activation_fct = "GELU"
  cfg.network.BosonNet.use_distance_attention_bias = (
      DEFAULTS.use_distance_attention_bias)
  cfg.network.BosonNet.distance_attention_bias_num_rbf = (
      DEFAULTS.distance_attention_bias_num_rbf)
  cfg.network.BosonNet.distance_attention_bias_eps = (
      DEFAULTS.distance_attention_bias_eps)
  cfg.network.BosonNet.distance_attention_bias_scale = (
      DEFAULTS.distance_attention_bias_scale)
  cfg.network.BosonNet.use_dipole_attention_bias = (
      DEFAULTS.use_dipole_attention_bias)
  cfg.network.BosonNet.dipole_attention_bias_scale = (
      DEFAULTS.dipole_attention_bias_scale)
  cfg.network.BosonNet.dipole_strength = dipole_strength


def _configure_mcmc(
    cfg: ml_collections.ConfigDict,
    *,
    lattice: np.ndarray,
) -> None:
  box_width = float(np.min(np.linalg.norm(lattice, axis=0)))

  cfg.mcmc.init_layout = DEFAULTS.init_layout
  cfg.mcmc.init_width = DEFAULTS.init_width_fraction * box_width
  cfg.mcmc.move_width = DEFAULTS.move_width
  cfg.mcmc.global_move_fraction = DEFAULTS.global_move_fraction
  cfg.mcmc.global_width_scale = DEFAULTS.global_width_scale
  cfg.mcmc.adapt_frequency = DEFAULTS.adapt_frequency
  cfg.mcmc.adapt_rate = DEFAULTS.adapt_rate
  cfg.mcmc.min_move_width = DEFAULTS.min_move_width
  cfg.mcmc.max_move_width = box_width


def _configure_runtime(
    cfg: ml_collections.ConfigDict,
    *,
    training_stages: list[dict],
    restore_path: str,
    reset_iteration_on_restore: bool,
    reset_optimizer_on_restore: bool,
    best_checkpoint_std_weight: float,
    batch_size: int,
    seed: int,
) -> None:
  cfg.batch_size = batch_size
  cfg.training.stages = training_stages

  cfg.optim.iterations = sum(stage["iterations"] for stage in training_stages[1:])
  cfg.optim.reset_optimizer_on_restore = reset_optimizer_on_restore

  cfg.log.stats_frequency = DEFAULTS.stats_frequency
  cfg.log.save_frequency = DEFAULTS.save_frequency_minutes
  cfg.log.restore_path = restore_path
  cfg.log.reset_iteration_on_restore = reset_iteration_on_restore
  cfg.log.best_checkpoint_min_step = DEFAULTS.best_checkpoint_min_step
  cfg.log.best_checkpoint_metric = DEFAULTS.best_checkpoint_metric
  cfg.log.best_checkpoint_std_weight = best_checkpoint_std_weight

  cfg.debug.deterministic = True
  cfg.debug.seed = seed
  cfg.debug.check_initial_energy = False


def build_config(
    *,
    num_bosons: int = DEFAULTS.num_bosons,
    layer_occupations: Sequence[int] = DEFAULTS.layer_occupations,
    layer_separation: float = DEFAULTS.layer_separation,
    dipole_strength: float = DEFAULTS.dipole_strength,
    supercell_shape: str = DEFAULTS.supercell_shape,
    density_rs: float = DEFAULTS.density_rs,
    seed: int = DEFAULTS.seed,
    burn_in_iterations: int = DEFAULTS.burn_in_iterations,
    fine_iterations: int = DEFAULTS.fine_iterations,
    results_dir: str | Path = DEFAULTS.results_dir,
    restore_path: str = DEFAULTS.restore_path,
    reset_iteration_on_restore: bool = DEFAULTS.reset_iteration_on_restore,
    reset_optimizer_on_restore: bool = DEFAULTS.reset_optimizer_on_restore,
    adiabatic_inter_start: float | None = DEFAULTS.adiabatic_inter_start,
    adiabatic_intra_start: float | None = DEFAULTS.adiabatic_intra_start,
    adiabatic_num_stages: int = DEFAULTS.adiabatic_num_stages,
    adiabatic_schedule: str = DEFAULTS.adiabatic_schedule,
    adiabatic_intra_final: float = DEFAULTS.adiabatic_intra_final,
    adiabatic_inter_final: float = DEFAULTS.adiabatic_inter_final,
    adiabatic_iterations: int = DEFAULTS.adiabatic_iterations,
    adiabatic_stage_lr_rate: float = DEFAULTS.adiabatic_stage_lr_rate,
    fine_lr_rate: float = DEFAULTS.fine_lr_rate,
    adiabatic_start_target: float = DEFAULTS.adiabatic_start_target,
    best_checkpoint_std_weight: float = DEFAULTS.best_checkpoint_std_weight,
    batch_size: int = DEFAULTS.batch_size,
) -> tuple[ml_collections.ConfigDict, np.ndarray]:
  """Builds the bilayer VMC config and fixed layer labels.

  This function is intentionally side-effect free: it does not write files,
  print device information, or start training. Use `scripts/train/run_bilayer.py`
  as the executable entry point for one run.
  """
  layer_occupations = _as_layer_occupations(layer_occupations)
  if sum(layer_occupations) != num_bosons:
    raise ValueError("layer_occupations must sum to num_bosons.")

  default_intra_start, default_inter_start = _physics_adiabatic_start_scales(
      density_rs=density_rs,
      layer_separation=layer_separation,
      dipole_strength=dipole_strength,
      target=adiabatic_start_target)
  if adiabatic_intra_start is None:
    adiabatic_intra_start = default_intra_start
  if adiabatic_inter_start is None:
    adiabatic_inter_start = default_inter_start

  warmup = AdiabaticWarmup(
      inter_start=adiabatic_inter_start,
      intra_start=adiabatic_intra_start,
      num_stages=adiabatic_num_stages,
      schedule=adiabatic_schedule,
      intra_final=adiabatic_intra_final,
      inter_final=adiabatic_inter_final,
      iterations=adiabatic_iterations,
      lr_rate=adiabatic_stage_lr_rate)
  training_stages = _make_training_stages(
      burn_in_iterations=burn_in_iterations,
      fine_iterations=fine_iterations,
      fine_lr_rate=fine_lr_rate,
      warmup=warmup)

  lattice = _make_lattice(
      num_bosons=num_bosons,
      density_rs=density_rs,
      supercell_shape=supercell_shape)

  cfg = default_config.default()
  _configure_system(
      cfg,
      num_bosons=num_bosons,
      layer_separation=layer_separation,
      dipole_strength=dipole_strength,
      lattice=lattice)
  _configure_network(cfg, dipole_strength=dipole_strength)
  _configure_mcmc(cfg, lattice=lattice)
  _configure_runtime(
      cfg,
      training_stages=training_stages,
      restore_path=restore_path,
      reset_iteration_on_restore=reset_iteration_on_restore,
      reset_optimizer_on_restore=reset_optimizer_on_restore,
      best_checkpoint_std_weight=best_checkpoint_std_weight,
      batch_size=batch_size,
      seed=seed)

  cfg.log.save_path = _result_folder(
      results_dir=results_dir,
      num_bosons=num_bosons,
      layer_occupations=layer_occupations,
      density_rs=density_rs,
      layer_separation=layer_separation,
      dipole_strength=dipole_strength,
      seed=seed,
      supercell_shape=supercell_shape)

  return cfg, _make_layer_assignment(layer_occupations)
