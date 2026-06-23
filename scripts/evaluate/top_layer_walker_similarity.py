#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Analyze bilayer sixfold and twelvefold bond-orientation sectors."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import re
import sys
import types

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
  rs = float(parsed["rs"])
  lat_vec = _lattice(num_bosons, rs, parsed["cell"])
  return {
      "num_bosons": num_bosons,
      "layer_occupations": (int(parsed["layer_a"]), int(parsed["layer_b"])),
      "rs": rs,
      "d": float(parsed["d"]),
      "dipole_strength": float(parsed["dipole"]),
      "seed": int(parsed["seed"]) if parsed["seed"] is not None else None,
      "supercell_shape": parsed["cell"],
      "lat_vec": lat_vec,
      "rec": 2 * np.pi * np.linalg.inv(lat_vec),
  }


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
  for pattern in ("qmcjax_ckpt_*.npz", "qmcjax_best_*.npz"):
    for path in run_dir.glob(pattern):
      try:
        files.append((int(path.stem.split("_")[-1]), path))
      except ValueError:
        pass
    if files:
      break
  if not files:
    raise FileNotFoundError(f"No qmcjax checkpoint found in {run_dir}")
  return [path for _, path in sorted(files, reverse=True)[:load_n_ckpts]]


def _fold_positions(positions: np.ndarray, lat_vec: np.ndarray,
                    rec: np.ndarray) -> np.ndarray:
  return np.asarray([
      lattices.send_positions_to_first_unit_cell(config, lat_vec, rec)
      for config in positions
  ])


def _load_walkers(run_dir: Path, checkpoint: Path | None,
                  load_n_ckpts: int, max_configs: int | None,
                  params: dict) -> tuple[np.ndarray, np.ndarray]:
  _install_jax_array_unpickle_fallback()
  position_chunks = []
  spin_chunks = []
  ckpt_files = _checkpoint_files(run_dir, checkpoint, load_n_ckpts)
  print(f"Loading {[path.name for path in ckpt_files]}")
  for ckpt_file in ckpt_files:
    ckpt = np.load(ckpt_file, allow_pickle=True)
    data = ckpt["data"].item()
    position_chunks.append(np.asarray(data["positions"]))
    spin_chunks.append(np.asarray(data["spins"]))

  positions = np.asarray(position_chunks).reshape(
      (-1, params["num_bosons"], 2))
  spins = np.asarray(spin_chunks).reshape((-1, params["num_bosons"]))
  if max_configs is not None and positions.shape[0] > max_configs:
    indices = np.linspace(0, positions.shape[0] - 1, max_configs, dtype=int)
    positions = positions[indices]
    spins = spins[indices]
  positions = _fold_positions(positions, params["lat_vec"], params["rec"])
  return positions, spins


def _layer_positions(positions: np.ndarray, spins: np.ndarray,
                     layer_sign: float) -> np.ndarray:
  layer_counts = np.sum(spins == layer_sign, axis=1)
  if not np.all(layer_counts == layer_counts[0]):
    raise ValueError("Layer particle count changes across walkers.")
  if layer_counts[0] == 0:
    raise ValueError(f"No layer-sign {layer_sign:g} particles found.")
  layer_positions = []
  for config, labels in zip(positions, spins):
    layer_positions.append(config[labels == layer_sign])
  return np.asarray(layer_positions)


def _minimum_image(lat_vec: np.ndarray, displacements: np.ndarray) -> np.ndarray:
  frac = np.einsum("ij,...j->...i", np.linalg.inv(lat_vec), displacements)
  frac = (frac + 0.5) % 1.0 - 0.5
  return np.einsum("ij,...j->...i", lat_vec, frac)


def _nearest_bond_indices(
    positions: np.ndarray,
    lat_vec: np.ndarray,
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
  num_particles = positions.shape[0]
  if num_particles < 2:
    raise ValueError("Need at least two particles in the layer.")
  neighbor_count = min(neighbor_count, num_particles - 1)
  disp = _minimum_image(lat_vec, positions[:, None, :] - positions[None, :, :])
  dist2 = np.sum(disp ** 2, axis=-1)
  np.fill_diagonal(dist2, np.inf)
  edges = set()
  for i in range(num_particles):
    neighbors = np.argpartition(dist2[i], neighbor_count)[:neighbor_count]
    for j in neighbors:
      a, b = sorted((int(i), int(j)))
      edges.add((a, b))
  if not edges:
    raise ValueError("No nearest-neighbor bonds found.")
  src, dst = zip(*sorted(edges))
  return np.asarray(src), np.asarray(dst)


def _bond_angles(
    positions: np.ndarray,
    lat_vec: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
  src, dst = _nearest_bond_indices(positions, lat_vec, neighbor_count)
  disp = _minimum_image(lat_vec, positions[dst] - positions[src])
  return np.arctan2(disp[:, 1], disp[:, 0])


def _order_parameter(angles: np.ndarray, harmonic: int) -> complex:
  return np.mean(np.exp(1j * harmonic * angles))


def _orientation_phase(psi: complex, harmonic: int) -> float:
  period = 2.0 * np.pi / harmonic
  return (np.angle(psi) / harmonic) % period


def _analyze_layer_orientations(
    layer_positions: np.ndarray,
    lat_vec: np.ndarray,
    neighbor_count: int,
    prefix: str,
) -> list[dict]:
  rows = []
  for walker_index, positions in enumerate(layer_positions):
    angles = _bond_angles(positions, lat_vec, neighbor_count)
    psi6 = _order_parameter(angles, 6)
    psi12 = _order_parameter(angles, 12)
    rows.append({
        "walker_index": walker_index,
        f"{prefix}_num_bonds": int(angles.size),
        f"{prefix}_psi6_real": float(np.real(psi6)),
        f"{prefix}_psi6_imag": float(np.imag(psi6)),
        f"{prefix}_psi6_abs": float(np.abs(psi6)),
        f"{prefix}_phi6_rad": float(_orientation_phase(psi6, 6)),
        f"{prefix}_phi6_deg": float(np.degrees(_orientation_phase(psi6, 6))),
        f"{prefix}_psi12_real": float(np.real(psi12)),
        f"{prefix}_psi12_imag": float(np.imag(psi12)),
        f"{prefix}_psi12_abs": float(np.abs(psi12)),
        f"{prefix}_phi12_rad": float(_orientation_phase(psi12, 12)),
        f"{prefix}_phi12_deg": float(np.degrees(_orientation_phase(psi12, 12))),
    })
  return rows


def _wrapped_delta_deg(values: np.ndarray, period: float) -> np.ndarray:
  return (values + 0.5 * period) % period - 0.5 * period


def _merge_layer_rows(top_rows: list[dict],
                      bottom_rows: list[dict]) -> list[dict]:
  if len(top_rows) != len(bottom_rows):
    raise ValueError("Top and bottom row counts differ.")
  rows = []
  for top, bottom in zip(top_rows, bottom_rows):
    if top["walker_index"] != bottom["walker_index"]:
      raise ValueError("Top and bottom walker indices differ.")
    row = dict(top)
    row.update(bottom)
    delta6 = _wrapped_delta_deg(
        np.asarray([row["bottom_phi6_deg"] - row["top_phi6_deg"]]), 60.0)[0]
    delta12 = _wrapped_delta_deg(
        np.asarray([row["bottom_phi12_deg"] - row["top_phi12_deg"]]), 30.0)[0]
    row["delta_phi6_deg"] = float(delta6)
    row["delta_phi12_deg"] = float(delta12)
    row["combined_psi6_weight"] = (
        row["top_psi6_abs"] * row["bottom_psi6_abs"])
    row["combined_psi12_weight"] = (
        row["top_psi12_abs"] * row["bottom_psi12_abs"])
    rows.append(row)
  return rows


def _save_csv(run_dir: Path, rows: list[dict], suffix: str) -> Path:
  path = run_dir / f"bilayer_bond_orientation{suffix}.csv"
  fieldnames = [
      "walker_index",
      "delta_phi6_deg",
      "delta_phi12_deg",
      "combined_psi6_weight",
      "combined_psi12_weight",
  ]
  for prefix in ("top", "bottom"):
    fieldnames.extend([
        f"{prefix}_num_bonds",
        f"{prefix}_psi6_real",
        f"{prefix}_psi6_imag",
        f"{prefix}_psi6_abs",
        f"{prefix}_phi6_rad",
        f"{prefix}_phi6_deg",
        f"{prefix}_psi12_real",
        f"{prefix}_psi12_imag",
        f"{prefix}_psi12_abs",
        f"{prefix}_phi12_rad",
        f"{prefix}_phi12_deg",
    ])
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  return path


def _save_plots(run_dir: Path, rows: list[dict], bins: int,
                suffix: str) -> Path:
  top_phi6 = np.asarray([row["top_phi6_deg"] for row in rows])
  bottom_phi6 = np.asarray([row["bottom_phi6_deg"] for row in rows])
  delta_phi6 = np.asarray([row["delta_phi6_deg"] for row in rows])
  top_abs6 = np.asarray([row["top_psi6_abs"] for row in rows])
  bottom_abs6 = np.asarray([row["bottom_psi6_abs"] for row in rows])
  top_abs12 = np.asarray([row["top_psi12_abs"] for row in rows])
  bottom_abs12 = np.asarray([row["bottom_psi12_abs"] for row in rows])
  combined6 = np.asarray([row["combined_psi6_weight"] for row in rows])

  fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  axes[0, 0].hist(top_phi6, bins=bins, range=(0.0, 60.0),
                  weights=top_abs6, color="#3b6ea8", edgecolor="white")
  axes[0, 0].set_xlabel("phi6 modulo 60 deg")
  axes[0, 0].set_ylabel("sum |psi6|")
  axes[0, 0].set_title("Top sixfold orientation")

  axes[0, 1].hist(bottom_phi6, bins=bins, range=(0.0, 60.0),
                  weights=bottom_abs6, color="#8f5aa3", edgecolor="white")
  axes[0, 1].set_xlabel("phi6 modulo 60 deg")
  axes[0, 1].set_ylabel("sum |psi6|")
  axes[0, 1].set_title("Bottom sixfold orientation")

  axes[0, 2].hist(delta_phi6, bins=bins, range=(-30.0, 30.0),
                  weights=combined6, color="#4c8c4a", edgecolor="white")
  axes[0, 2].set_xlabel("bottom - top phi6, deg")
  axes[0, 2].set_ylabel("sum |psi6_top| |psi6_bottom|")
  axes[0, 2].set_title("Relative sixfold orientation")

  scatter = axes[1, 0].scatter(
      top_phi6,
      bottom_phi6,
      c=combined6,
      s=16,
      alpha=0.75,
      cmap="viridis")
  axes[1, 0].set_xlabel("top phi6 modulo 60 deg")
  axes[1, 0].set_ylabel("bottom phi6 modulo 60 deg")
  axes[1, 0].set_title("Top vs bottom sectors")
  fig.colorbar(scatter, ax=axes[1, 0], label="|psi6_top| |psi6_bottom|")

  axes[1, 1].scatter(
      top_abs6,
      bottom_abs6,
      s=14,
      alpha=0.75,
      color="#c44e52")
  axes[1, 1].set_xlabel("top |psi6|")
  axes[1, 1].set_ylabel("bottom |psi6|")
  axes[1, 1].set_title("Sixfold strength")

  axes[1, 2].scatter(
      top_abs12,
      bottom_abs12,
      s=14,
      alpha=0.75,
      color="#6d6d6d")
  axes[1, 2].set_xlabel("top |psi12|")
  axes[1, 2].set_ylabel("bottom |psi12|")
  axes[1, 2].set_title("Twelvefold strength")

  fig.tight_layout()
  path = run_dir / f"fig_bilayer_bond_orientation{suffix}.png"
  fig.savefig(path, dpi=200)
  plt.close(fig)
  return path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--load-n-ckpts", type=int, default=1)
  parser.add_argument("--max-configs", type=int, default=0)
  parser.add_argument("--neighbor-count", type=int, default=6)
  parser.add_argument("--bins", type=int, default=36)
  parser.add_argument("--output-suffix", default="")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  params = _parse_run_dir(args.run_dir)
  max_configs = None if args.max_configs == 0 else args.max_configs
  positions, spins = _load_walkers(
      args.run_dir,
      args.checkpoint,
      args.load_n_ckpts,
      max_configs,
      params)
  top_positions = _layer_positions(positions, spins, 1.0)
  bottom_positions = _layer_positions(positions, spins, -1.0)
  top_rows = _analyze_layer_orientations(
      top_positions,
      params["lat_vec"],
      args.neighbor_count,
      "top")
  bottom_rows = _analyze_layer_orientations(
      bottom_positions,
      params["lat_vec"],
      args.neighbor_count,
      "bottom")
  rows = _merge_layer_rows(top_rows, bottom_rows)
  csv_path = _save_csv(args.run_dir, rows, args.output_suffix)
  fig_path = _save_plots(args.run_dir, rows, args.bins, args.output_suffix)

  top_psi6_abs = np.asarray([row["top_psi6_abs"] for row in rows])
  bottom_psi6_abs = np.asarray([row["bottom_psi6_abs"] for row in rows])
  top_psi12_abs = np.asarray([row["top_psi12_abs"] for row in rows])
  bottom_psi12_abs = np.asarray([row["bottom_psi12_abs"] for row in rows])
  delta_phi6 = np.asarray([row["delta_phi6_deg"] for row in rows])
  print(f"Top-layer particles: {top_positions.shape[1]}")
  print(f"Bottom-layer particles: {bottom_positions.shape[1]}")
  print(f"Walkers analyzed: {len(rows)}")
  print(f"Neighbor count: {args.neighbor_count}")
  print(
      f"top |psi6| mean={np.mean(top_psi6_abs):.6g}, "
      f"std={np.std(top_psi6_abs):.6g}")
  print(
      f"bottom |psi6| mean={np.mean(bottom_psi6_abs):.6g}, "
      f"std={np.std(bottom_psi6_abs):.6g}")
  print(
      f"top |psi12| mean={np.mean(top_psi12_abs):.6g}, "
      f"std={np.std(top_psi12_abs):.6g}")
  print(
      f"bottom |psi12| mean={np.mean(bottom_psi12_abs):.6g}, "
      f"std={np.std(bottom_psi12_abs):.6g}")
  print(
      f"delta_phi6 mean={np.mean(delta_phi6):.6g} deg, "
      f"std={np.std(delta_phi6):.6g} deg")
  print(f"Saved {csv_path}")
  print(f"Saved {fig_path}")


if __name__ == "__main__":
  main()
