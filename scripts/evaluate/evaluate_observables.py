#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Evaluate density and structure factor for bilayer boson scan runs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs("/tmp/matplotlib", exist_ok=True)
os.makedirs("/tmp/fontconfig", exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from periodicwave.pbc import lattices


DEFAULT_SCAN_DIR = (
    REPO_ROOT
    / "results"
    / "bilayer-bosons"
    / "BosonNet"
    / "scan_260603"
)
DEFAULT_PATTERN = "N*_layers*_*_rs*_d*_D*_sq"
RUN_RE = re.compile(
    r"N(?P<num_bosons>\d+)_layers(?P<layer_a>\d+)_(?P<layer_b>\d+)"
    r"_rs(?P<rs>[0-9.]+)_d(?P<d>[0-9.]+)_D(?P<dipole>[0-9.]+)"
    r"(?:_seed(?P<seed>\d+))?_"
    r"(?P<cell>sq|tri)(?:_[^/]+)?$"
)


@dataclass(frozen=True)
class RunParams:
  path: Path
  num_bosons: int
  layer_occupations: tuple[int, int]
  rs: float
  d: float
  dipole_strength: float
  seed: int | None
  supercell_shape: str
  lat_vec: np.ndarray
  rec: np.ndarray
  layer_assignment: np.ndarray


def _install_jax_array_unpickle_fallback():
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
    return lat_vec
  if supercell_shape == "sq":
    supercell_a = rs * np.sqrt(np.pi * num_bosons)
    return lattices._square_lattice_vecs(supercell_a)
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")


def _parse_run_dir(run_dir: Path) -> RunParams:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  parsed = match.groupdict()
  num_bosons = int(parsed["num_bosons"])
  layer_occupations = (int(parsed["layer_a"]), int(parsed["layer_b"]))
  rs = float(parsed["rs"])
  supercell_shape = parsed["cell"]
  lat_vec = _lattice(num_bosons, rs, supercell_shape)
  rec = 2 * np.pi * np.linalg.inv(lat_vec)
  layer_assignment = np.array(
      [1.0] * layer_occupations[0] + [-1.0] * layer_occupations[1])
  return RunParams(
      path=run_dir,
      num_bosons=num_bosons,
      layer_occupations=layer_occupations,
      rs=rs,
      d=float(parsed["d"]),
      dipole_strength=float(parsed["dipole"]),
      seed=int(parsed["seed"]) if parsed["seed"] is not None else None,
      supercell_shape=supercell_shape,
      lat_vec=lat_vec,
      rec=rec,
      layer_assignment=layer_assignment,
  )


def _format_run_title(params: RunParams) -> str:
  parts = [f"rs={params.rs:g}", f"d={params.d:g}"]
  if params.seed is not None:
    parts.append(f"seed={params.seed}")
  return ", ".join(parts)


def _latest_checkpoint_files(folder_path: Path, nfiles: int) -> list[Path]:
  numbered = []
  for path in folder_path.glob("qmcjax_ckpt_*.npz"):
    try:
      numbered.append((int(path.stem.split("_")[-1]), path))
    except ValueError:
      pass
  return [path for _, path in sorted(numbered, reverse=True)[:nfiles]]


def _best_checkpoint_files(folder_path: Path, nfiles: int) -> list[Path]:
  manifest_path = folder_path / "qmcjax_best_checkpoints.csv"
  if not manifest_path.exists() or manifest_path.stat().st_size == 0:
    return []
  rows = np.genfromtxt(
      manifest_path,
      delimiter=",",
      names=True,
      dtype=None,
      encoding="utf-8")
  rows = np.atleast_1d(rows)
  if rows.size == 0:
    return []

  candidates = []
  for row in rows:
    try:
      score = float(row["score"])
      checkpoint_name = str(row["checkpoint"])
    except (ValueError, KeyError):
      continue
    checkpoint_path = folder_path / checkpoint_name
    if checkpoint_path.exists():
      candidates.append((score, checkpoint_path))
  candidates.sort(key=lambda item: item[0])
  return [path for _, path in candidates[:nfiles]]


def _checkpoint_files(folder_path: Path, nfiles: int) -> list[Path]:
  best_files = _best_checkpoint_files(folder_path, nfiles)
  if best_files:
    return best_files
  return _latest_checkpoint_files(folder_path, nfiles)


def _explicit_checkpoint_files(run_dir: Path, checkpoint: Path) -> list[Path]:
  checkpoint = checkpoint.expanduser()
  if not checkpoint.is_absolute():
    checkpoint = run_dir / checkpoint
  checkpoint = checkpoint.resolve()
  if not checkpoint.exists():
    raise ValueError(f"--checkpoint does not exist: {checkpoint}")
  if not checkpoint.is_file():
    raise ValueError(f"--checkpoint is not a file: {checkpoint}")
  return [checkpoint]


def _load_positions(
    params: RunParams,
    nfiles: int,
    max_configs: int | None,
    checkpoint: Path | None = None,
) -> np.ndarray:
  _install_jax_array_unpickle_fallback()
  positions = []
  ckpt_files = (
      _explicit_checkpoint_files(params.path, checkpoint)
      if checkpoint is not None
      else _checkpoint_files(params.path, nfiles))
  if not ckpt_files:
    raise ValueError(f"No checkpoints found in {params.path}")
  print(
      f"Loading checkpoints for {params.path.name}: "
      f"{[path.name for path in ckpt_files]}")
  for ckpt_file in ckpt_files:
    ckpt = np.load(ckpt_file, allow_pickle=True)
    data = ckpt["data"].item()
    positions.append(np.asarray(data["positions"]))
  arr = np.asarray(positions).reshape((-1, params.num_bosons, 2))
  if max_configs is not None and arr.shape[0] > max_configs:
    indices = np.linspace(0, arr.shape[0] - 1, max_configs, dtype=int)
    arr = arr[indices]
  return np.asarray([
      lattices.send_positions_to_first_unit_cell(config, params.lat_vec, params.rec)
      for config in arr
  ])


def _cell_outline(params: RunParams) -> np.ndarray:
  corners_frac = np.array([
      [-0.5, -0.5],
      [0.5, -0.5],
      [0.5, 0.5],
      [-0.5, 0.5],
      [-0.5, -0.5],
  ])
  return np.einsum("ij,kj->ki", params.lat_vec, corners_frac)


def _minimum_image(params: RunParams, displacements: np.ndarray) -> np.ndarray:
  frac = np.einsum("ij,...j->...i", np.linalg.inv(params.lat_vec), displacements)
  frac = (frac + 0.5) % 1.0 - 0.5
  return np.einsum("ij,...j->...i", params.lat_vec, frac)


def _scatter_density(
    ax,
    params: RunParams,
    positions: np.ndarray,
    title: str,
    color: str,
) -> None:
  flat = positions.reshape((-1, 2))
  npoints = flat.shape[0]
  point_size = np.clip(1800 / max(npoints, 1), 1.5, 8.0)

  ax.scatter(
      flat[:, 0],
      flat[:, 1],
      s=point_size,
      alpha=0.7,
      color=color,
      linewidths=0,
      rasterized=True)

  outline = _cell_outline(params)
  ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.0, alpha=0.65)
  pad = 0.04 * np.max(np.ptp(outline, axis=0))
  ax.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
  ax.set_ylim(outline[:, 1].min() - pad, outline[:, 1].max() + pad)
  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title(f"{title} ({npoints} samples)")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)


def _draw_cell(ax, params: RunParams) -> None:
  outline = _cell_outline(params)
  ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.0, alpha=0.65)
  pad = 0.04 * np.max(np.ptp(outline, axis=0))
  ax.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
  ax.set_ylim(outline[:, 1].min() - pad, outline[:, 1].max() + pad)
  ax.set_aspect("equal", adjustable="box")
  ax.set_xticks([])
  ax.set_yticks([])


def _save_snapshot_plots(
    params: RunParams,
    positions: np.ndarray,
    snapshot_count: int,
) -> None:
  nsnapshots = min(snapshot_count, positions.shape[0])
  if nsnapshots == 0:
    return
  indices = np.linspace(0, positions.shape[0] - 1, nsnapshots, dtype=int)
  ncols = min(4, nsnapshots)
  nrows = int(np.ceil(nsnapshots / ncols))
  fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
  axs = np.atleast_1d(axs).ravel()
  colors = np.where(params.layer_assignment == 1.0, "#c7364f", "#2f7d57")
  for ax, idx in zip(axs, indices):
    config = positions[idx]
    ax.scatter(
        config[:, 0],
        config[:, 1],
        s=32,
        c=colors,
        edgecolors="black",
        linewidths=0.4)
    _draw_cell(ax, params)
    ax.set_title(f"sample {idx}")
  for ax in axs[len(indices):]:
    ax.axis("off")
  fig.suptitle(_format_run_title(params), y=0.995)
  fig.tight_layout()
  path = params.path / "fig_positions_xy_snapshots.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def _compute_pair_correlation(
    params: RunParams,
    positions: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
  area = abs(np.linalg.det(params.lat_vec))
  density = params.num_bosons / area
  lengths = np.linalg.norm(params.lat_vec, axis=0)
  rmax = 0.5 * np.min(lengths)
  edges = np.linspace(0.0, rmax, nbins + 1)
  counts = np.zeros(nbins)
  for config in positions:
    disp = config[:, None, :] - config[None, :, :]
    disp = _minimum_image(params, disp)
    distances = np.linalg.norm(disp, axis=-1)
    pair_distances = distances[np.triu_indices(params.num_bosons, k=1)]
    counts += np.histogram(pair_distances, bins=edges)[0]
  centers = 0.5 * (edges[:-1] + edges[1:])
  shell_areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
  ideal_counts = positions.shape[0] * params.num_bosons * density * shell_areas / 2.0
  gr = np.divide(
      counts,
      ideal_counts,
      out=np.zeros_like(counts, dtype=float),
      where=ideal_counts > 0)
  return centers, gr


def _pair_correlation_edges(
    params: RunParams,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  lengths = np.linalg.norm(params.lat_vec, axis=0)
  rmax = 0.5 * np.min(lengths)
  edges = np.linspace(0.0, rmax, nbins + 1)
  centers = 0.5 * (edges[:-1] + edges[1:])
  shell_areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
  return edges, centers, shell_areas


def _compute_layer_pair_correlation(
    params: RunParams,
    positions: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
  area = abs(np.linalg.det(params.lat_vec))
  edges, centers, shell_areas = _pair_correlation_edges(params, nbins)
  counts = np.zeros(nbins)
  same_layer = np.array_equal(mask_a, mask_b)

  for config in positions:
    pos_a = config[mask_a]
    pos_b = config[mask_b]
    disp = pos_a[:, None, :] - pos_b[None, :, :]
    disp = _minimum_image(params, disp)
    distances = np.linalg.norm(disp, axis=-1)
    if same_layer:
      pair_distances = distances[np.triu_indices(pos_a.shape[0], k=1)]
    else:
      pair_distances = distances.ravel()
    counts += np.histogram(pair_distances, bins=edges)[0]

  density_b = np.count_nonzero(mask_b) / area
  ideal_counts = positions.shape[0] * np.count_nonzero(mask_a) * density_b * shell_areas
  if same_layer:
    ideal_counts /= 2.0
  gr = np.divide(
      counts,
      ideal_counts,
      out=np.zeros_like(counts, dtype=float),
      where=ideal_counts > 0)
  return centers, gr


def _pair_correlation_xy_edges(
    params: RunParams,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
  outline = _cell_outline(params)
  xmin, xmax = outline[:, 0].min(), outline[:, 0].max()
  ymin, ymax = outline[:, 1].min(), outline[:, 1].max()
  return np.linspace(xmin, xmax, nbins + 1), np.linspace(ymin, ymax, nbins + 1)


def _compute_layer_pair_correlation_xy(
    params: RunParams,
    positions: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  area = abs(np.linalg.det(params.lat_vec))
  xedges, yedges = _pair_correlation_xy_edges(params, nbins)
  counts = np.zeros((nbins, nbins))
  same_layer = np.array_equal(mask_a, mask_b)

  for config in positions:
    pos_a = config[mask_a]
    pos_b = config[mask_b]
    disp = pos_a[:, None, :] - pos_b[None, :, :]
    disp = _minimum_image(params, disp)
    if same_layer:
      disp = disp[~np.eye(pos_a.shape[0], dtype=bool)]
    else:
      disp = disp.reshape((-1, 2))
    counts += np.histogram2d(
        disp[:, 0],
        disp[:, 1],
        bins=(xedges, yedges))[0]

  bin_area = np.diff(xedges)[:, None] * np.diff(yedges)[None, :]
  density_b = np.count_nonzero(mask_b) / area
  ideal_counts = (
      positions.shape[0] * np.count_nonzero(mask_a) * density_b * bin_area)

  gxy = np.divide(
      counts,
      ideal_counts,
      out=np.full_like(counts, np.nan, dtype=float),
      where=ideal_counts > 0)

  # For non-rectangular cells, the Cartesian histogram bounds include corners
  # outside the primitive-cell parallelogram used by the minimum-image map.
  xcenters = 0.5 * (xedges[:-1] + xedges[1:])
  ycenters = 0.5 * (yedges[:-1] + yedges[1:])
  xx, yy = np.meshgrid(xcenters, ycenters, indexing="ij")
  frac = np.einsum(
      "ij,...j->...i",
      np.linalg.inv(params.lat_vec),
      np.stack([xx, yy], axis=-1))
  inside_cell = np.all((frac >= -0.5) & (frac < 0.5), axis=-1)
  gxy[~inside_cell] = np.nan
  return xedges, yedges, gxy


def _save_pair_correlation(
    params: RunParams,
    positions: np.ndarray,
    nbins: int,
) -> None:
  r, gr = _compute_pair_correlation(params, positions, nbins)
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  ax.plot(r, gr, linewidth=1.5)
  ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
  ax.set_xlabel("r")
  ax.set_ylabel("g(r)")
  ax.set_title(f"Pair correlation: {_format_run_title(params)}")
  ax.grid(alpha=0.25, linewidth=0.5)
  fig.tight_layout()
  path = params.path / "fig_pair_correlation_gr.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def _save_layer_pair_correlations(
    params: RunParams,
    positions: np.ndarray,
    nbins: int,
) -> None:
  top = params.layer_assignment == 1.0
  bottom = params.layer_assignment == -1.0
  curves = [
      ("top-top", "#c7364f", top, top),
      ("bottom-bottom", "#2f7d57", bottom, bottom),
      ("top-bottom", "#2a6fbb", top, bottom),
  ]

  fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
  csv_columns = []
  for label, color, mask_a, mask_b in curves:
    r, gr = _compute_layer_pair_correlation(
        params, positions, mask_a, mask_b, nbins)
    ax = axs[1] if label == "top-bottom" else axs[0]
    ax.plot(r, gr, linewidth=1.5, label=label, color=color)
    csv_columns.append((label, gr))

  titles = ["same-layer", "top-bottom"]
  for ax, title in zip(axs, titles):
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("in-plane r")
    ax.set_ylabel("g(r)")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()
  fig.suptitle(
      f"Layer-resolved pair correlation: {_format_run_title(params)}")
  fig.tight_layout()
  path = params.path / "fig_pair_correlation_gr_by_layer.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)

  data = np.column_stack([r] + [gr for _, gr in csv_columns])
  header = "r," + ",".join(label.replace("-", "_") for label, _ in csv_columns)
  csv_path = params.path / "pair_correlation_gr_by_layer.csv"
  np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
  print(f"Saved {csv_path}")


def _save_layer_pair_correlation_xy(
    params: RunParams,
    positions: np.ndarray,
    nbins: int,
) -> None:
  top = params.layer_assignment == 1.0
  bottom = params.layer_assignment == -1.0
  curves = [
      ("top-top", top, top),
      ("bottom-bottom", bottom, bottom),
      ("top-bottom", top, bottom),
  ]

  results = []
  for label, mask_a, mask_b in curves:
    xedges, yedges, gxy = _compute_layer_pair_correlation_xy(
        params, positions, mask_a, mask_b, nbins)
    results.append((label, xedges, yedges, gxy))

  finite_arrays = [
      gxy[np.isfinite(gxy)].ravel()
      for _, _, _, gxy in results
      if np.any(np.isfinite(gxy))
  ]
  finite_values = (
      np.concatenate(finite_arrays) if finite_arrays else np.asarray([1.0]))
  vmax = float(np.percentile(finite_values, 99.0)) if finite_values.size else 1.0
  vmax = max(vmax, 1.0)

  fig, axs = plt.subplots(
      1,
      len(results),
      figsize=(4.2 * len(results), 4.2),
      constrained_layout=True)
  axs = np.atleast_1d(axs)
  mesh = None
  for ax, (label, xedges, yedges, gxy) in zip(axs, results):
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        gxy.T,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x separation")
    ax.set_ylabel("y separation")
    ax.set_title(label)
  if mesh is not None:
    fig.colorbar(mesh, ax=axs, shrink=0.86, label="g(x, y)")
  fig.suptitle(
      f"Layer-resolved pair correlation g(x, y): {_format_run_title(params)}")
  path = params.path / "fig_pair_correlation_gxy_by_layer.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def _save_density_plots(
    params: RunParams,
    positions: np.ndarray,
    write_extra_figures: bool = True,
) -> None:
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  _scatter_density(ax, params, positions, "Overall xy density", color="#2a6fbb")
  fig.suptitle(_format_run_title(params), y=0.995)
  fig.tight_layout()
  path = params.path / "fig_density_xy_overall.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)

  fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
  masks = [params.layer_assignment == 1.0, params.layer_assignment == -1.0]
  titles = ["top layer", "bottom layer"]
  colors = ["#c7364f", "#2f7d57"]
  for ax, mask, title, color in zip(axs, masks, titles, colors):
    _scatter_density(ax, params, positions[:, mask, :], title, color)
  fig.suptitle(f"Layer densities: {_format_run_title(params)}", y=0.995)
  fig.tight_layout()
  path = params.path / "fig_density_xy_by_layer.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)

  if not write_extra_figures:
    return

  fig, ax = plt.subplots(1, 1, figsize=(5, 4))
  zs = [-0.5 * params.d, 0.5 * params.d]
  counts = [
      np.sum(params.layer_assignment == -1.0),
      np.sum(params.layer_assignment == 1.0),
  ]
  ax.bar(zs, counts, width=0.2 * params.d)
  ax.set_xlabel("z")
  ax.set_ylabel("particle count")
  ax.set_title(f"Density along z: {_format_run_title(params)}")
  fig.tight_layout()
  path = params.path / "fig_density_z_layers.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def _compute_structure_factor(
    params: RunParams,
    positions: np.ndarray,
    kmax: int,
    normalization: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  if normalization is None:
    normalization = positions.shape[1]
  ms = []
  vals = []
  for m1 in range(-kmax, kmax + 1):
    for m2 in range(-kmax, kmax + 1):
      if m1 == 0 and m2 == 0:
        continue
      kvec = params.rec @ np.array([m1, m2])
      phases = np.exp(1j * np.einsum("cnd,d->cn", positions, kvec))
      rho_k = np.sum(phases, axis=1)
      sk = np.mean(np.abs(rho_k) ** 2) / normalization
      ms.append((m1, m2))
      vals.append(sk)
  return np.asarray(ms), np.asarray(vals)


def _save_structure_factor(
    params: RunParams,
    positions: np.ndarray,
    kmax: int,
) -> None:
  ms, vals = _compute_structure_factor(params, positions, kmax)
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  scatter = ax.scatter(ms[:, 0], ms[:, 1], c=vals, s=80, cmap="viridis")
  fig.colorbar(scatter, ax=ax, label="S(k)")
  ax.set_xlabel("m1")
  ax.set_ylabel("m2")
  ax.set_title(f"Static structure factor: {_format_run_title(params)}")
  ax.set_aspect("equal", adjustable="box")
  fig.tight_layout()
  path = params.path / "fig_structure_factor_sk.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def _save_layer_structure_factors(
    params: RunParams,
    positions: np.ndarray,
    kmax: int,
) -> None:
  layers = [
      ("top", params.layer_assignment == 1.0),
      ("bottom", params.layer_assignment == -1.0),
  ]
  layer_results = []
  for label, mask in layers:
    layer_positions = positions[:, mask, :]
    ms, vals = _compute_structure_factor(
        params,
        layer_positions,
        kmax,
        normalization=layer_positions.shape[1])
    layer_results.append((label, ms, vals))

  vmax = max(float(np.max(vals)) for _, _, vals in layer_results)
  fig, axs = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
  for ax, (label, ms, vals) in zip(axs, layer_results):
    scatter = ax.scatter(
        ms[:, 0],
        ms[:, 1],
        c=vals,
        s=80,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax)
    ax.set_xlabel("m1")
    ax.set_ylabel("m2")
    ax.set_title(f"{label} layer S(k)")
    ax.set_aspect("equal", adjustable="box")
  fig.colorbar(scatter, ax=axs, shrink=0.88, label="S_layer(k)")
  fig.suptitle(
      f"Layer static structure factors: {_format_run_title(params)}")
  path = params.path / "fig_structure_factor_sk_by_layer.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def _plot_structure_factor_panel(
    fig,
    ax,
    ms: np.ndarray,
    vals: np.ndarray,
    title: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
  scatter = ax.scatter(
      ms[:, 0],
      ms[:, 1],
      c=vals,
      s=48,
      cmap="viridis",
      vmin=vmin,
      vmax=vmax)
  fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=label)
  ax.set_xlabel("m1")
  ax.set_ylabel("m2")
  ax.set_title(title)
  ax.set_aspect("equal", adjustable="box")


def _save_density_structure_overview(
    params: RunParams,
    positions: np.ndarray,
    kmax: int,
) -> None:
  masks = [
      ("overall", positions, "#2a6fbb"),
      ("top layer", positions[:, params.layer_assignment == 1.0, :], "#c7364f"),
      ("bottom layer", positions[:, params.layer_assignment == -1.0, :], "#2f7d57"),
  ]
  overall_ms, overall_sk = _compute_structure_factor(params, positions, kmax)

  layer_sk = []
  for label, layer_positions, _ in masks[1:]:
    ms, vals = _compute_structure_factor(
        params,
        layer_positions,
        kmax,
        normalization=layer_positions.shape[1])
    layer_sk.append((label, ms, vals))
  layer_vmax = max(float(np.max(vals)) for _, _, vals in layer_sk)

  fig, axs = plt.subplots(
      2,
      3,
      figsize=(14, 8.2),
      constrained_layout=True)
  for ax, (label, panel_positions, color) in zip(axs[0], masks):
    _scatter_density(ax, params, panel_positions, f"{label} density", color)

  _plot_structure_factor_panel(
      fig,
      axs[1, 0],
      overall_ms,
      overall_sk,
      "overall S(k)",
      "S(k)")
  for ax, (label, ms, vals) in zip(axs[1, 1:], layer_sk):
    _plot_structure_factor_panel(
        fig,
        ax,
        ms,
        vals,
        f"{label} S(k)",
        "S_layer(k)",
        vmin=0.0,
        vmax=layer_vmax)

  fig.suptitle(
      f"Density and structure overview: {_format_run_title(params)}")
  path = params.path / "fig_density_structure_overview.png"
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def _outputs_exist(run_dir: Path, write_extra_figures: bool = True) -> bool:
  names = [
      "fig_density_structure_overview.png",
      "fig_pair_correlation_gr_by_layer.png",
      "fig_pair_correlation_gxy_by_layer.png",
  ]
  if write_extra_figures:
    names.extend([
        "fig_density_xy_overall.png",
        "fig_density_xy_by_layer.png",
        "fig_density_z_layers.png",
        "fig_positions_xy_snapshots.png",
        "fig_pair_correlation_gr.png",
        "fig_structure_factor_sk.png",
        "fig_structure_factor_sk_by_layer.png",
    ])
  return all((run_dir / name).exists() for name in names)


def _evaluate_run(
    run_dir: Path,
    load_n_ckpts: int,
    max_configs: int | None,
    snapshot_count: int,
    pair_correlation_bins: int,
    kmax: int,
    skip_existing: bool,
    write_extra_figures: bool = True,
    checkpoint: Path | None = None,
) -> None:
  if skip_existing and _outputs_exist(run_dir, write_extra_figures):
    print(f"Skipping existing bilayer plots: {run_dir}")
    return

  params = _parse_run_dir(run_dir)
  positions = _load_positions(params, load_n_ckpts, max_configs, checkpoint)
  _save_density_structure_overview(params, positions, kmax)
  if write_extra_figures:
    _save_density_plots(params, positions, write_extra_figures)
    _save_snapshot_plots(params, positions, snapshot_count)
  if write_extra_figures:
    _save_pair_correlation(params, positions, pair_correlation_bins)
  _save_layer_pair_correlations(params, positions, pair_correlation_bins)
  _save_layer_pair_correlation_xy(params, positions, pair_correlation_bins)
  if write_extra_figures:
    _save_structure_factor(params, positions, kmax)
    _save_layer_structure_factors(params, positions, kmax)


def _select_run_dirs(run_dir: Path | None, scan_dir: Path, pattern: str) -> list[Path]:
  if run_dir is not None:
    selected = run_dir.resolve()
    if not selected.is_dir():
      raise ValueError(f"--run-dir does not exist or is not a directory: {selected}")
    return [selected]

  scan_root = scan_dir.resolve()
  run_dirs = sorted(path for path in scan_root.glob(pattern) if path.is_dir())
  if not run_dirs:
    raise ValueError(f"No run directories matched {scan_root / pattern}")
  return run_dirs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--run-dir",
      type=Path,
      default=None,
      help="Evaluate one run directory directly; ignores --scan-dir and --pattern.",
  )
  parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
  parser.add_argument("--pattern", default=DEFAULT_PATTERN)
  parser.add_argument("--load-n-ckpts", type=int, default=1)
  parser.add_argument(
      "--checkpoint",
      type=Path,
      default=None,
      help=(
          "Evaluate this checkpoint file instead of selecting best/latest "
          "checkpoints. Relative paths are resolved inside --run-dir."
      ),
  )
  parser.add_argument(
      "--max-configs",
      type=int,
      default=0,
      help="Subsample configurations per run; use 0 for all.",
  )
  parser.add_argument("--snapshot-count", type=int, default=12)
  parser.add_argument("--pair-correlation-bins", type=int, default=80)
  parser.add_argument("--kmax", type=int, default=5)
  parser.add_argument(
      "--skip-existing",
      action="store_true",
      help="Skip runs where the requested bilayer output files already exist.",
  )
  parser.add_argument(
      "--write-extra-figures",
      action="store_true",
      help=(
          "Also write separate density, snapshot, total g(r), and separate "
          "S(k) figures."
      ),
  )
  args = parser.parse_args()

  max_configs = None if args.max_configs == 0 else args.max_configs
  run_dirs = _select_run_dirs(args.run_dir, args.scan_dir, args.pattern)

  failures = []
  for idx, run_dir in enumerate(run_dirs, start=1):
    print(f"Evaluating bilayer {idx}/{len(run_dirs)}: {run_dir.name}", flush=True)
    try:
      _evaluate_run(
          run_dir,
          args.load_n_ckpts,
          max_configs,
          args.snapshot_count,
          args.pair_correlation_bins,
          args.kmax,
          args.skip_existing,
          args.write_extra_figures,
          args.checkpoint,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failures.append((run_dir, exc))
      print(f"Failed {run_dir.name}: {exc}", flush=True)

  print(f"Processed {len(run_dirs) - len(failures)}/{len(run_dirs)} bilayer runs")
  if failures:
    print(f"Failures: {len(failures)}")
    for path, exc in failures[:10]:
      print(f"  {path.name}: {exc}")


if __name__ == "__main__":
  main()
