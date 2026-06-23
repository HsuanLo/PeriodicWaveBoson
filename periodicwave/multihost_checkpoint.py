"""Multi-process checkpoint helpers for JAX distributed runs."""

from __future__ import annotations

import os
from typing import Optional

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from periodicwave import network_interfaces as networks
from periodicwave.checkpoint import prune_checkpoints as _core_prune_checkpoints
from periodicwave.checkpoint import save as _core_save
from periodicwave.checkpoint import update_best_checkpoints as _core_update_best


def _rank0() -> bool:
  return jax.process_index() == 0


def _gather_walkers(value):
  from jax.experimental.multihost_utils import process_allgather

  gathered = np.asarray(process_allgather(np.asarray(value)))
  shape = gathered.shape
  total_walkers = shape[0] * shape[1] * shape[2]
  return gathered.reshape(1, total_walkers, *shape[3:])


def _split_walkers(value, walkers_per_process: int, local_devices: int):
  value = np.asarray(value)
  total_walkers = value.shape[0] * value.shape[1]
  if total_walkers % jax.process_count() != 0:
    raise ValueError(
        "Checkpoint walker field is not divisible by process count, got "
        f"{total_walkers} walkers for {jax.process_count()} processes.")
  if total_walkers // jax.process_count() != walkers_per_process:
    raise ValueError("Checkpoint walker fields have inconsistent sizes.")

  flat = value.reshape(total_walkers, *value.shape[2:])
  start = jax.process_index() * walkers_per_process
  local = flat[start:start + walkers_per_process]
  return local.reshape(
      local_devices, walkers_per_process // local_devices, *value.shape[2:])


def save(
    save_path: str,
    t: int,
    data: networks.WalkerData,
    params,
    opt_state,
    mcmc_width,
) -> str:
  """Saves a full-worker checkpoint from a distributed JAX process set."""
  ckpt_filename = os.path.join(save_path, f"qmcjax_ckpt_{t:06d}.npz")
  if jax.process_count() <= 1:
    return _core_save(save_path, t, data, params, opt_state, mcmc_width)

  global_data = data.replace(
      positions=_gather_walkers(data.positions),
      spins=_gather_walkers(data.spins),
      atoms=_gather_walkers(data.atoms),
      charges=_gather_walkers(data.charges))

  if _rank0():
    _core_save(save_path, t, global_data, params, opt_state, mcmc_width)
  return ckpt_filename


def restore(restore_filename: str, batch_size: Optional[int] = None):
  """Restores a checkpoint, splitting aggregated walkers across processes."""
  logging.info("Loading checkpoint %s", restore_filename)
  with open(restore_filename, "rb") as f:
    ckpt_data = np.load(f, allow_pickle=True)
    t = ckpt_data["t"].tolist() + 1
    data = networks.WalkerData(**ckpt_data["data"].item())
    params = ckpt_data["params"].tolist()
    opt_state = ckpt_data["opt_state"].tolist()
    mcmc_width = jnp.array(ckpt_data["mcmc_width"].tolist())

  process_count = jax.process_count()
  local_devices = jax.local_device_count()
  stored_positions = np.asarray(data.positions)
  if stored_positions.shape[0] == local_devices and process_count == 1:
    if batch_size is not None:
      actual = stored_positions.shape[0] * stored_positions.shape[1]
      if actual != batch_size:
        raise ValueError(
            f"Wrong batch size in loaded data. Expected {batch_size}, "
            f"found {actual}.")
    return t, data, params, opt_state, mcmc_width

  total_walkers = stored_positions.shape[0] * stored_positions.shape[1]
  if total_walkers % process_count != 0:
    raise ValueError(
        "Checkpoint walker count must be divisible by process count, got "
        f"{total_walkers} walkers for {process_count} processes.")

  walkers_per_process = total_walkers // process_count
  if walkers_per_process % local_devices != 0:
    raise ValueError(
        "Per-process walker count must be divisible by local devices, got "
        f"{walkers_per_process} walkers for {local_devices} devices.")

  data = data.replace(
      positions=_split_walkers(
          data.positions, walkers_per_process, local_devices),
      spins=_split_walkers(data.spins, walkers_per_process, local_devices),
      atoms=_split_walkers(data.atoms, walkers_per_process, local_devices),
      charges=_split_walkers(data.charges, walkers_per_process, local_devices))

  if batch_size is not None:
    actual = data.positions.shape[0] * data.positions.shape[1]
    if actual != batch_size:
      raise ValueError(
          f"Wrong local batch size in loaded data. Expected {batch_size}, "
          f"found {actual}.")

  return t, data, params, opt_state, mcmc_width


def prune_checkpoints(ckpt_path: str, keep_latest: int) -> None:
  if _rank0():
    _core_prune_checkpoints(ckpt_path, keep_latest)


def update_best_checkpoints(
    ckpt_filename: str,
    ckpt_path: str,
    step: int,
    score: float,
    keep_best: int,
) -> None:
  if _rank0():
    _core_update_best(ckpt_filename, ckpt_path, step, score, keep_best)
