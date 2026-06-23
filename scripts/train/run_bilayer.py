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


def _distributed_preinit() -> None:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--coordinator_address", default=None)
  parser.add_argument("--process_index", type=int, default=None)
  parser.add_argument("--local_device_ids", default="0")
  parser.add_argument("--num_processes", type=int, default=1)
  args, _ = parser.parse_known_args()
  if args.num_processes <= 1:
    return

  import jax as _jax

  process_index = args.process_index
  if process_index is None:
    process_index = int(os.environ.get("SLURM_PROCID", 0))
  local_device_ids = [
      int(device_id)
      for device_id in args.local_device_ids.split(",")
      if device_id.strip()
  ]
  _jax.distributed.initialize(
      coordinator_address=args.coordinator_address,
      num_processes=args.num_processes,
      process_id=process_index,
      local_device_ids=local_device_ids)


_distributed_preinit()

from periodicwave.configs import bilayer_bosons


DEFAULTS = bilayer_bosons.DEFAULTS


class _NullWriter:
  """No-op train writer for non-rank-0 distributed processes."""

  class _NullFile:

    def flush(self) -> None:
      pass

  _file = _NullFile()

  def __enter__(self):
    return self

  def __exit__(self, *args) -> None:
    pass

  def write(self, t: int, **kwargs) -> None:
    pass


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
  parser.add_argument(
      "--init-layout",
      choices=("jittered_lattice", "farthest"),
      default=_env_str("BILAYER_INIT_LAYOUT", DEFAULTS.init_layout))
  parser.add_argument("--burn-in-iterations", type=int,
                      default=_env_int(
                          "BILAYER_BURN_IN_ITERATIONS",
                          DEFAULTS.burn_in_iterations))
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
  parser.add_argument("--adiabatic-inter-start", type=float,
                      default=_env_float(
                          "BILAYER_ADIABATIC_INTER_START",
                          DEFAULTS.adiabatic_inter_start))
  parser.add_argument("--adiabatic-intra-start", type=float,
                      default=_env_float(
                          "BILAYER_ADIABATIC_INTRA_START",
                          DEFAULTS.adiabatic_intra_start))
  parser.add_argument("--adiabatic-num-stages", type=int,
                      default=_env_int(
                          "BILAYER_ADIABATIC_NUM_STAGES",
                          DEFAULTS.adiabatic_num_stages))
  parser.add_argument("--adiabatic-schedule",
                      choices=("log", "linear"),
                      default=_env_str(
                          "BILAYER_ADIABATIC_SCHEDULE",
                          DEFAULTS.adiabatic_schedule))
  parser.add_argument("--adiabatic-intra-final", type=float,
                      default=_env_float(
                          "BILAYER_ADIABATIC_INTRA_FINAL",
                          DEFAULTS.adiabatic_intra_final))
  parser.add_argument("--adiabatic-inter-final", type=float,
                      default=_env_float(
                          "BILAYER_ADIABATIC_INTER_FINAL",
                          DEFAULTS.adiabatic_inter_final))
  parser.add_argument("--adiabatic-iterations", type=int,
                      default=_env_int(
                          "BILAYER_ADIABATIC_ITERATIONS",
                          DEFAULTS.adiabatic_iterations))
  parser.add_argument("--adiabatic-stage-lr-rate", type=float,
                      default=_env_float(
                          "BILAYER_ADIABATIC_STAGE_LR_RATE",
                          DEFAULTS.adiabatic_stage_lr_rate))
  parser.add_argument("--fine-lr-rate", type=float,
                      default=_env_float(
                          "BILAYER_FINE_LR_RATE",
                          DEFAULTS.fine_lr_rate))
  parser.add_argument("--adiabatic-start-target", type=float,
                      default=_env_float(
                          "BILAYER_ADIABATIC_START_TARGET",
                          DEFAULTS.adiabatic_start_target))
  parser.add_argument("--best-checkpoint-std-weight", type=float,
                      default=_env_float(
                          "BEST_CHECKPOINT_STD_WEIGHT",
                          DEFAULTS.best_checkpoint_std_weight))
  parser.add_argument("--batch-size", type=int,
                      default=_env_int("BILAYER_BATCH_SIZE",
                                       DEFAULTS.batch_size))
  mn = parser.add_argument_group("distributed")
  mn.add_argument("--coordinator_address", default=None)
  mn.add_argument("--process_index", type=int, default=None)
  mn.add_argument("--local_device_ids", default="0")
  mn.add_argument("--num_processes", type=int, default=1)
  mn.add_argument("--jobid", default=_env_str("SLURM_JOB_ID", ""))
  return parser.parse_args()


def run(args: argparse.Namespace) -> None:
  from absl import logging
  import jax
  from jax.experimental import multihost_utils

  from periodicwave import train
  from periodicwave.utils import custom_logging
  from periodicwave.utils import writers

  jax.config.update("jax_default_matmul_precision", "float32")
  is_rank0 = jax.process_index() == 0
  if is_rank0:
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
      fine_iterations=args.fine_iterations,
      results_dir=args.results_dir,
      restore_path=args.restore_path,
      reset_iteration_on_restore=args.reset_iteration_on_restore,
      reset_optimizer_on_restore=args.reset_optimizer_on_restore,
      adiabatic_inter_start=args.adiabatic_inter_start,
      adiabatic_intra_start=args.adiabatic_intra_start,
      adiabatic_num_stages=args.adiabatic_num_stages,
      adiabatic_schedule=args.adiabatic_schedule,
      adiabatic_intra_final=args.adiabatic_intra_final,
      adiabatic_inter_final=args.adiabatic_inter_final,
      adiabatic_iterations=args.adiabatic_iterations,
      adiabatic_stage_lr_rate=args.adiabatic_stage_lr_rate,
      fine_lr_rate=args.fine_lr_rate,
      adiabatic_start_target=args.adiabatic_start_target,
      best_checkpoint_std_weight=args.best_checkpoint_std_weight,
      batch_size=args.batch_size,
  )
  cfg.mcmc.init_layout = args.init_layout
  if args.jobid and args.num_processes > 1:
    cfg.log.save_path = f"{cfg.log.save_path}_jobid{args.jobid}"

  if is_rank0:
    os.makedirs(cfg.log.save_path, exist_ok=True)
  if jax.process_count() > 1:
    multihost_utils.sync_global_devices("periodicwave_save_path_ready")

  from periodicwave import checkpoint as core_checkpoint
  from periodicwave import multihost_checkpoint

  core_checkpoint.save = multihost_checkpoint.save
  core_checkpoint.restore = multihost_checkpoint.restore
  core_checkpoint.prune_checkpoints = multihost_checkpoint.prune_checkpoints
  core_checkpoint.update_best_checkpoints = (
      multihost_checkpoint.update_best_checkpoints)

  if is_rank0:
    writers.rename_file("device_info", cfg.log.save_path, file_extension="log")
    custom_logging.log_device_info(cfg.log.save_path + "/device_info.log")
    writers.rename_file("config", cfg.log.save_path, file_extension="json")
    custom_logging.save_config_dict_as_json(
        cfg, cfg.log.save_path + "/config.json")

  t_init = time()
  writer_manager = None if is_rank0 else _NullWriter()
  train.train(
      cfg,
      writer_manager=writer_manager,
      layer_assignment=layer_assignment)
  if is_rank0:
    logging.info("Training completed after t [s] = %d", int(time() - t_init))


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
