#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot pair-center static structure factor from bilayer checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import re
import sys
import types

import numpy as np


DEBUG_ROOT = Path(__file__).resolve().parents[2]
if str(DEBUG_ROOT) in sys.path:
  sys.path.remove(str(DEBUG_ROOT))
sys.path.insert(0, str(DEBUG_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from periodicwave.pbc import lattices


RUN_RE = re.compile(
    r"N(?P<num_bosons>\d+)_layers(?P<layer_a>\d+)_(?P<layer_b>\d+)"
    r"_rs(?P<rs>[0-9.]+)_d(?P<d>[0-9.]+)_D(?P<dipole>[0-9.]+)"
    r"(?:_seed(?P<seed>\d+))?_"
    r"(?P<cell>sq|tri)(?:_[^/]+)?$")


def _install_jax_array_unpickle_fallback() -> None:
  """Let NumPy checkpoints containing JAX arrays load as plain ndarrays."""
  if "jax._src.array" in sys.modules:
    return
  if importlib.util.find_spec("jax") is not None:
    return

  def _reconstruct_array(reconstruct_func, reconstruct_args, state, metadata):
    del metadata
    arr = reconstruct_func(*reconstruct_args)
    arr.__setstate__(state)
    return arr

  jax_module = types.ModuleType("jax")
  jax_src_module = types.ModuleType("jax._src")
  jax_array_module = types.ModuleType("jax._src.array")
  jax_array_module._reconstruct_array = _reconstruct_array
  sys.modules.setdefault("jax", jax_module)
  sys.modules.setdefault("jax._src", jax_src_module)
  sys.modules.setdefault("jax._src.array", jax_array_module)


def _lattice(num_bosons: int, rs: float, supercell_shape: str) -> np.ndarray:
  if supercell_shape == "tri":
    supercell_a = rs * np.sqrt(2 * np.pi / np.sqrt(3) * num_bosons)
    lat_vec, _ = lattices._triangular_lattice_vecs_periodic_potential(
        supercell_a, 1)
    return np.asarray(lat_vec)
  if supercell_shape == "sq":
    supercell_a = rs * np.sqrt(np.pi * num_bosons)
    return np.asarray(lattices._square_lattice_vecs(supercell_a))
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")


def _parse_run_dir(run_dir: Path) -> dict:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  parsed = match.groupdict()
  num_bosons = int(parsed["num_bosons"])
  layer_occupations = (int(parsed["layer_a"]), int(parsed["layer_b"]))
  if layer_occupations[0] != layer_occupations[1]:
    raise ValueError(
        "Pair-center structure currently expects equal layer occupations.")
  rs = float(parsed["rs"])
  lat_vec = _lattice(num_bosons, rs, parsed["cell"])
  return {
      "num_bosons": num_bosons,
      "num_pairs": layer_occupations[0],
      "layer_occupations": layer_occupations,
      "rs": rs,
      "d": float(parsed["d"]),
      "dipole_strength": float(parsed["dipole"]),
      "seed": int(parsed["seed"]) if parsed["seed"] is not None else None,
      "supercell_shape": parsed["cell"],
      "lat_vec": lat_vec,
      "rec": 2 * np.pi * np.linalg.inv(lat_vec),
  }


def _minimum_image(lat_vec: np.ndarray, displacements: np.ndarray) -> np.ndarray:
  frac = np.einsum("ij,...j->...i", np.linalg.inv(lat_vec), displacements)
  frac = (frac + 0.5) % 1.0 - 0.5
  return np.einsum("ij,...j->...i", lat_vec, frac)


def _fold_positions(positions: np.ndarray, lat_vec: np.ndarray,
                    rec: np.ndarray) -> np.ndarray:
  return np.asarray([
      lattices.send_positions_to_first_unit_cell(config, lat_vec, rec)
      for config in positions
  ])


def _checkpoint_files(run_dir: Path, checkpoint: Path | None,
                      load_n_ckpts: int) -> list[Path]:
  if checkpoint is not None:
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_absolute():
      checkpoint = run_dir / checkpoint
    if not checkpoint.exists():
      raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return [checkpoint]

  files = []
  for path in run_dir.glob("qmcjax_ckpt_*.npz"):
    try:
      files.append((int(path.stem.split("_")[-1]), path))
    except ValueError:
      pass
  if not files:
    for path in run_dir.glob("qmcjax_best_*.npz"):
      try:
        files.append((int(path.stem.split("_")[-1]), path))
      except ValueError:
        pass
  if not files:
    raise FileNotFoundError(f"No qmcjax checkpoint found in {run_dir}")
  return [path for _, path in sorted(files, reverse=True)[:load_n_ckpts]]


def _load_positions(run_dir: Path, checkpoint: Path | None,
                    load_n_ckpts: int, max_configs: int | None,
                    params: dict) -> np.ndarray:
  _install_jax_array_unpickle_fallback()
  chunks = []
  ckpt_files = _checkpoint_files(run_dir, checkpoint, load_n_ckpts)
  print(f"Loading {[path.name for path in ckpt_files]}")
  for ckpt_file in ckpt_files:
    ckpt = np.load(ckpt_file, allow_pickle=True)
    data = ckpt["data"].item()
    chunks.append(np.asarray(data["positions"]))
  positions = np.asarray(chunks).reshape((-1, params["num_bosons"], 2))
  if max_configs is not None and positions.shape[0] > max_configs:
    indices = np.linspace(0, positions.shape[0] - 1, max_configs, dtype=int)
    positions = positions[indices]
  return _fold_positions(positions, params["lat_vec"], params["rec"])


def _pair_centers(positions: np.ndarray, params: dict) -> np.ndarray:
  num_pairs = params["num_pairs"]
  top = positions[:, :num_pairs, :]
  bottom = positions[:, num_pairs:2 * num_pairs, :]
  rel = _minimum_image(params["lat_vec"], top - bottom)
  centers = bottom + 0.5 * rel
  return _fold_positions(centers, params["lat_vec"], params["rec"])


def _compute_structure_factor(
    pair_centers: np.ndarray,
    params: dict,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray]:
  ms = []
  vals = []
  for m1 in range(-kmax, kmax + 1):
    for m2 in range(-kmax, kmax + 1):
      if m1 == 0 and m2 == 0:
        continue
      kvec = params["rec"] @ np.array([m1, m2])
      phases = np.exp(1j * np.einsum("cnd,d->cn", pair_centers, kvec))
      rho_k = np.sum(phases, axis=1)
      sk = np.mean(np.abs(rho_k) ** 2) / params["num_pairs"]
      ms.append((m1, m2))
      vals.append(sk)
  return np.asarray(ms), np.asarray(vals)


def _format_title(params: dict) -> str:
  parts = [f"rs={params['rs']:g}", f"d={params['d']:g}"]
  if params["seed"] is not None:
    parts.append(f"seed={params['seed']}")
  return ", ".join(parts)


def _save_csv(run_dir: Path, ms: np.ndarray, vals: np.ndarray,
              suffix: str) -> Path:
  path = run_dir / f"pair_center_structure_factor{suffix}.csv"
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["m1", "m2", "S_pair"])
    for (m1, m2), val in zip(ms, vals):
      writer.writerow([int(m1), int(m2), float(val)])
  return path


def _save_plot(run_dir: Path, ms: np.ndarray, vals: np.ndarray,
               params: dict, suffix: str) -> Path:
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  scatter = ax.scatter(
      ms[:, 0],
      ms[:, 1],
      c=vals,
      s=90,
      cmap="viridis")
  fig.colorbar(scatter, ax=ax, label="S_pair(k)")
  ax.set_xlabel("m1")
  ax.set_ylabel("m2")
  ax.set_title(f"Pair-center static structure factor: {_format_title(params)}")
  ax.set_aspect("equal", adjustable="box")
  fig.tight_layout()
  path = run_dir / f"fig_pair_center_structure_factor{suffix}.png"
  fig.savefig(path, dpi=200)
  plt.close(fig)
  return path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--load-n-ckpts", type=int, default=1)
  parser.add_argument("--max-configs", type=int, default=0)
  parser.add_argument("--kmax", type=int, default=5)
  parser.add_argument("--output-suffix", default="")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  params = _parse_run_dir(args.run_dir)
  max_configs = None if args.max_configs == 0 else args.max_configs
  positions = _load_positions(
      args.run_dir,
      args.checkpoint,
      args.load_n_ckpts,
      max_configs,
      params)
  centers = _pair_centers(positions, params)
  ms, vals = _compute_structure_factor(centers, params, args.kmax)
  csv_path = _save_csv(args.run_dir, ms, vals, args.output_suffix)
  fig_path = _save_plot(args.run_dir, ms, vals, params, args.output_suffix)
  peak = int(np.argmax(vals))
  print(f"Saved {csv_path}")
  print(f"Saved {fig_path}")
  print(
      "Peak S_pair(k): "
      f"m=({int(ms[peak, 0])}, {int(ms[peak, 1])}), "
      f"S_pair={float(vals[peak]):.8g}, "
      f"S_pair/N_pair={float(vals[peak] / params['num_pairs']):.8g}")


if __name__ == "__main__":
  main()
