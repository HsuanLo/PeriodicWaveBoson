#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Run one bilayer boson NN-VMC calculation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import time

from periodicwave.configs import bilayer_bosons


DEFAULTS = bilayer_bosons.DEFAULTS


def _env_float(name: str, default: float) -> float:
  value = os.environ.get(name)
  return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
  value = os.environ.get(name)
  return default if value is None else int(value)


def _env_bool(name: str, default: bool) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.lower() in ("1", "true", "yes", "y", "on")


def _env_str(name: str, default: str) -> str:
  value = os.environ.get(name)
  return default if value is None else value


def _parse_layers(value: str) -> tuple[int, int]:
  pieces = value.replace(",", "_").split("_")
  if len(pieces) != 2:
    raise argparse.ArgumentTypeError(
        "layers must have two entries, e.g. 12_12 or 12,12")
  return int(pieces[0]), int(pieces[1])


def _default_layers() -> tuple[int, int]:
  return _parse_layers(
      _env_str(
          "SCAN_LAYERS",
          _env_str(
              "ANNEAL_LAYERS",
              bilayer_bosons.format_layer_occupations(
                  DEFAULTS.layer_occupations))))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-bosons", type=int,
                      default=_env_int(
                          "SCAN_N", _env_int("ANNEAL_N", DEFAULTS.num_bosons)))
  parser.add_argument("--layers", type=_parse_layers, default=_default_layers())
  parser.add_argument("--rs", type=float,
                      default=_env_float("SCAN_RS", DEFAULTS.density_rs))
  parser.add_argument("--d", type=float,
                      default=_env_float(
                          "SCAN_D", DEFAULTS.layer_separation))
  parser.add_argument("--dipole-strength", type=float,
                      default=_env_float(
                          "SCAN_DIPOLE", DEFAULTS.dipole_strength))
  parser.add_argument("--cell",
                      default=_env_str("SCAN_CELL", DEFAULTS.supercell_shape))
  parser.add_argument("--seed", type=int,
                      default=_env_int("SCAN_SEED", DEFAULTS.seed))
  parser.add_argument("--burn-in-iterations", type=int,
                      default=_env_int(
                          "BILAYER_BURN_IN_ITERATIONS",
                          DEFAULTS.burn_in_iterations))
  parser.add_argument("--bold-iterations", type=int,
                      default=_env_int(
                          "BILAYER_BOLD_ITERATIONS",
                          DEFAULTS.bold_iterations))
  parser.add_argument("--retune-iterations", type=int,
                      default=_env_int(
                          "BILAYER_RETUNE_ITERATIONS",
                          DEFAULTS.retune_iterations))
  parser.add_argument("--fine-iterations", type=int,
                      default=_env_int(
                          "BILAYER_FINE_ITERATIONS",
                          DEFAULTS.fine_iterations))
  parser.add_argument("--results-dir", type=Path,
                      default=Path(_env_str(
                          "RESULTS_DIR", DEFAULTS.results_dir)))
  parser.add_argument("--restore-path",
                      default=_env_str("RESTORE_PATH", DEFAULTS.restore_path))
  parser.add_argument("--reset-iteration-on-restore", action="store_true",
                      default=_env_bool(
                          "RESET_ITERATION_ON_RESTORE",
                          DEFAULTS.reset_iteration_on_restore))
  parser.add_argument("--reset-optimizer-on-restore", action="store_true",
                      default=_env_bool(
                          "RESET_OPTIMIZER_ON_RESTORE",
                          DEFAULTS.reset_optimizer_on_restore))
  parser.add_argument("--best-checkpoint-std-weight", type=float,
                      default=_env_float(
                          "BEST_CHECKPOINT_STD_WEIGHT",
                          DEFAULTS.best_checkpoint_std_weight))
  return parser.parse_args()


def run(args: argparse.Namespace) -> None:
  from absl import logging
  import jax

  from periodicwave import train
  from periodicwave.utils import custom_logging
  from periodicwave.utils import writers

  jax.config.update("jax_default_matmul_precision", "float32")
  print("Jax Devices:", jax.devices())

  cfg, layer_assignment = bilayer_bosons.build_config(
      num_bosons=args.num_bosons,
      layer_occupations=args.layers,
      layer_separation=args.d,
      dipole_strength=args.dipole_strength,
      supercell_shape=args.cell,
      density_rs=args.rs,
      seed=args.seed,
      burn_in_iterations=args.burn_in_iterations,
      bold_iterations=args.bold_iterations,
      retune_iterations=args.retune_iterations,
      fine_iterations=args.fine_iterations,
      results_dir=args.results_dir,
      restore_path=args.restore_path,
      reset_iteration_on_restore=args.reset_iteration_on_restore,
      reset_optimizer_on_restore=args.reset_optimizer_on_restore,
      best_checkpoint_std_weight=args.best_checkpoint_std_weight,
  )

  writers.rename_file("device_info", cfg.log.save_path, file_extension="log")
  custom_logging.log_device_info(cfg.log.save_path + "/device_info.log")
  writers.rename_file("config", cfg.log.save_path, file_extension="json")
  custom_logging.save_config_dict_as_json(
      cfg, cfg.log.save_path + "/config.json")

  t_init = time()
  train.train(cfg, layer_assignment=layer_assignment)
  logging.info("Training completed after t [s] = %d", int(time() - t_init))


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
