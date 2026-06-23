#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Decompose bilayer kinetic energy into pair COM and relative pieces.

This is a diagnostic script for saved debug checkpoints. It uses fixed-index
top-bottom pairing: the kth +1 layer particle is paired with the kth -1 layer
particle in the checkpoint ordering. This convention is smooth under autodiff,
but it is a diagnostic convention rather than a permutation-invariant matching.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


DEBUG_ROOT = Path(__file__).resolve().parents[2]
if str(DEBUG_ROOT) in sys.path:
  sys.path.remove(str(DEBUG_ROOT))
sys.path.insert(0, str(DEBUG_ROOT))


def _parse_lattice(value) -> np.ndarray:
  """Parses a 2x2 lattice from JSON or an unserializable-object string."""
  if isinstance(value, list):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (2, 2):
      return arr
  if isinstance(value, str):
    nums = re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", value)
    vals = [float(num) for num in nums]
    if len(vals) >= 4:
      return np.asarray(vals[:4], dtype=np.float32).reshape(2, 2)
  raise ValueError(f"Could not parse 2x2 lattice from {value!r}")


def _latest_checkpoint(run_dir: Path) -> Path:
  checkpoints = sorted(run_dir.glob("qmcjax_ckpt_*.npz"))
  if not checkpoints:
    checkpoints = sorted(run_dir.glob("qmcjax_best_*.npz"))
  if not checkpoints:
    raise FileNotFoundError(f"No qmcjax checkpoint found in {run_dir}")
  return checkpoints[-1]


def _load_config(run_dir: Path) -> dict:
  with (run_dir / "config.json").open(encoding="utf-8") as f:
    return json.load(f)


def _load_checkpoint(path: Path):
  """Loads a checkpoint without enforcing the current device count."""
  from periodicwave import network_interfaces as networks

  with path.open("rb") as f:
    data = np.load(f, allow_pickle=True)
    step = int(data["t"].tolist())
    walkers = networks.WalkerData(**data["data"].item())
    params = data["params"].tolist()
  return step, walkers, params


def _tree_first_replica(tree, num_replicas: int):
  import jax
  import jax.numpy as jnp

  def first(x):
    x = jnp.asarray(x)
    if x.ndim > 0 and x.shape[0] == num_replicas:
      return x[0]
    return x

  return jax.tree_util.tree_map(first, tree)


def _flatten_walkers(data, max_walkers: int | None):
  import jax
  import jax.numpy as jnp
  from periodicwave import network_interfaces as networks

  flat = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, (-1,) + tuple(x.shape[2:])), data)
  if max_walkers is not None:
    flat = jax.tree_util.tree_map(lambda x: x[:max_walkers], flat)
  return networks.WalkerData(
      positions=flat.positions,
      spins=flat.spins,
      atoms=flat.atoms,
      charges=flat.charges,
  )


def _make_network(config: dict):
  import jax.numpy as jnp
  from periodicwave import BosonNet

  system = config["system"]
  network_cfg = config["network"]
  if network_cfg.get("network_type") != "BosonNet":
    raise ValueError("Only BosonNet checkpoints are supported.")

  lattice = _parse_lattice(system.get("pbc_lattice"))
  kwargs = dict(network_cfg["BosonNet"])
  return BosonNet.make_boson_net(
      tuple(system["bosons"]),
      jnp.asarray([0.0]),
      ndim=int(system.get("ndim", 2)),
      complex_output=bool(network_cfg.get("complex", False)),
      pbc_lattice=jnp.asarray(lattice),
      layer_separation=system["make_local_energy_kwargs"].get(
          "layer_separation", 1.0),
      **kwargs,
  )


def _fixed_pair_indices(spins):
  import jax.numpy as jnp

  spins_np = np.asarray(spins)
  top = np.where(spins_np > 0.0)[0]
  bottom = np.where(spins_np < 0.0)[0]
  if top.shape[0] != bottom.shape[0]:
    raise ValueError(
        f"Need equal top/bottom counts, got {top.shape[0]} and "
        f"{bottom.shape[0]}.")
  return jnp.asarray(top), jnp.asarray(bottom)


def _make_kinetic_evaluator(network, params, top_idx, bottom_idx,
                            include_cartesian_check: bool):
  import jax
  import jax.numpy as jnp

  num_pairs = int(top_idx.shape[0])
  q_cm_size = 2 * num_pairs

  def positions_to_q(positions):
    xy = jnp.reshape(positions, (-1, 2))
    top = xy[top_idx]
    bottom = xy[bottom_idx]
    centers = 0.5 * (top + bottom)
    rel = top - bottom
    return jnp.concatenate([centers.reshape(-1), rel.reshape(-1)])

  def q_to_positions(q, template_positions):
    centers = jnp.reshape(q[:q_cm_size], (num_pairs, 2))
    rel = jnp.reshape(q[q_cm_size:], (num_pairs, 2))
    top = centers + 0.5 * rel
    bottom = centers - 0.5 * rel
    xy = jnp.reshape(template_positions, (-1, 2))
    xy = xy.at[top_idx].set(top)
    xy = xy.at[bottom_idx].set(bottom)
    return xy.reshape(-1)

  def laplacian_parts(log_fn, x):
    grad_fn = jax.grad(log_fn)
    grad = grad_fn(x)
    eye = jnp.eye(x.shape[0], dtype=x.dtype)
    diag = []
    for i in range(x.shape[0]):
      diag.append(jax.jvp(grad_fn, (x,), (eye[i],))[1][i])
    return grad, jnp.stack(diag)

  def one_walker(positions, spins, atoms, charges):
    q = positions_to_q(positions)

    def log_q(q_in):
      return network.apply(
          params,
          q_to_positions(q_in, positions),
          spins,
          atoms,
          charges)[1]

    grad_q, diag_q = laplacian_parts(log_q, q)
    cm_lap = jnp.sum(diag_q[:q_cm_size])
    cm_grad2 = jnp.sum(grad_q[:q_cm_size] ** 2)
    rel_lap = jnp.sum(diag_q[q_cm_size:])
    rel_grad2 = jnp.sum(grad_q[q_cm_size:] ** 2)
    t_cm = -0.25 * (cm_lap + cm_grad2)
    t_rel = -(rel_lap + rel_grad2)

    if not include_cartesian_check:
      return t_cm, t_rel, jnp.nan

    def log_pos(pos_in):
      return network.apply(params, pos_in, spins, atoms, charges)[1]

    grad_pos, diag_pos = laplacian_parts(log_pos, positions)
    t_cart = -0.5 * (jnp.sum(diag_pos) + jnp.sum(grad_pos ** 2))
    return t_cm, t_rel, t_cart

  return jax.jit(jax.vmap(one_walker))


def _summary(values, num_particles: int):
  arr = np.asarray(values, dtype=np.float64)
  per_particle = arr / num_particles
  return {
      "mean": float(np.mean(per_particle)),
      "stderr": float(np.std(per_particle) / np.sqrt(per_particle.size)),
      "std": float(np.std(per_particle)),
  }


def _progress_bar(iterable, *, total: int, desc: str, disable: bool):
  """Wraps an iterable in tqdm when available."""
  if disable:
    return iterable
  try:
    from tqdm.auto import tqdm
  except ImportError:
    return iterable
  return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def _slice_walkers(data, start: int, end: int):
  from periodicwave import network_interfaces as networks

  return networks.WalkerData(
      positions=data.positions[start:end],
      spins=data.spins[start:end],
      atoms=data.atoms[start:end],
      charges=data.charges[start:end],
  )


def _evaluate_in_batches(evaluator, data, *, batch_size: int,
                         show_progress: bool):
  import jax

  nwalkers = int(data.positions.shape[0])
  if batch_size <= 0:
    batch_size = nwalkers
  starts = range(0, nwalkers, batch_size)
  total_batches = math.ceil(nwalkers / batch_size)
  t_cm_chunks = []
  t_rel_chunks = []
  t_cart_chunks = []
  for start in _progress_bar(
      starts,
      total=total_batches,
      desc="kinetic batches",
      disable=not show_progress):
    batch = _slice_walkers(data, start, min(start + batch_size, nwalkers))
    t_cm, t_rel, t_cart = evaluator(
        batch.positions, batch.spins, batch.atoms, batch.charges)
    t_cm_chunks.append(np.asarray(jax.device_get(t_cm)))
    t_rel_chunks.append(np.asarray(jax.device_get(t_rel)))
    t_cart_chunks.append(np.asarray(jax.device_get(t_cart)))
  return (
      np.concatenate(t_cm_chunks),
      np.concatenate(t_rel_chunks),
      np.concatenate(t_cart_chunks),
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, default=None,
                      help="Checkpoint filename/path. Defaults to latest ckpt.")
  parser.add_argument("--max-walkers", type=int, default=128,
                      help="Number of saved walkers to evaluate.")
  parser.add_argument("--batch-size", type=int, default=16,
                      help="Walkers per progress-bar batch. Use <=0 for all.")
  parser.add_argument("--no-progress", action="store_true",
                      help="Disable the tqdm progress bar.")
  parser.add_argument("--skip-cartesian-check", action="store_true",
                      help="Skip independent Cartesian kinetic recomputation.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  import jax

  config = _load_config(args.run_dir)
  checkpoint_path = (
      args.checkpoint
      if args.checkpoint is not None else _latest_checkpoint(args.run_dir))
  if not checkpoint_path.is_absolute():
    checkpoint_path = args.run_dir / checkpoint_path

  step, data, params = _load_checkpoint(checkpoint_path)
  num_replicas = int(data.positions.shape[0])
  params = _tree_first_replica(params, num_replicas)
  data = _flatten_walkers(data, args.max_walkers)
  network = _make_network(config)

  top_idx, bottom_idx = _fixed_pair_indices(data.spins[0])
  evaluator = _make_kinetic_evaluator(
      network,
      params,
      top_idx,
      bottom_idx,
      include_cartesian_check=not args.skip_cartesian_check)

  t_cm, t_rel, t_cart = _evaluate_in_batches(
      evaluator,
      data,
      batch_size=args.batch_size,
      show_progress=not args.no_progress)
  t_sum = t_cm + t_rel
  num_particles = int(data.spins.shape[1])
  nwalkers = int(data.positions.shape[0])

  result = {
      "run_dir": str(args.run_dir),
      "checkpoint": str(checkpoint_path),
      "checkpoint_step": step,
      "walkers_evaluated": nwalkers,
      "batch_size": args.batch_size,
      "num_particles": num_particles,
      "pairing": "fixed index: kth top-layer particle with kth bottom-layer particle",
      "T_cm_per_particle": _summary(t_cm, num_particles),
      "T_rel_per_particle": _summary(t_rel, num_particles),
      "T_cm_plus_T_rel_per_particle": _summary(t_sum, num_particles),
  }
  if not args.skip_cartesian_check:
    result["T_cartesian_per_particle"] = _summary(t_cart, num_particles)
    result["cartesian_minus_decomposed_per_particle"] = _summary(
        t_cart - t_sum, num_particles)

  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
