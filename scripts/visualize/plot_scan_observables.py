#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot density and structure factor across a bilayer boson scan."""

from __future__ import annotations

import argparse
import os
import sys
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
from matplotlib.colors import Normalize
import numpy as np

from periodicwave.pbc import lattices
import plot_scan_common as common


DEFAULT_PATTERN = common.DEFAULT_PATTERN


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


@dataclass(frozen=True)
class RunObservables:
  params: RunParams
  positions: np.ndarray
  structure_ms: np.ndarray
  structure_vals: np.ndarray
  pair_r: np.ndarray
  top_top_gr: np.ndarray
  bottom_bottom_gr: np.ndarray
  top_bottom_gr: np.ndarray


def _run_params(run_dir: Path) -> RunParams:
  parsed = common.parse_run_dir(run_dir)
  num_bosons = int(parsed["num_bosons"])
  layer_occupations = (int(parsed["layer_a"]), int(parsed["layer_b"]))
  rs = float(parsed["rs"])
  supercell_shape = parsed["cell"]
  lat_vec = common.lattice(num_bosons, rs, supercell_shape)
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


def _load_positions(
    params: RunParams,
    load_n_ckpts: int,
    max_configs: int | None,
) -> np.ndarray:
  common.install_jax_array_unpickle_fallback()
  positions = []
  ckpt_files = common.checkpoint_files(params.path, load_n_ckpts)
  if not ckpt_files:
    raise ValueError(f"No checkpoints found in {params.path}")
  print(
      f"Loading checkpoints for {params.path.name}: "
      f"{[path.name for path in ckpt_files]}")
  for ckpt_file in ckpt_files:
    ckpt = np.load(ckpt_file, allow_pickle=True)
    data = ckpt["data"].item()
    pos = np.asarray(data["positions"])
    positions.append(pos)

  arr = np.asarray(positions)
  arr = arr.reshape((-1, params.num_bosons, 2))
  if max_configs is not None and arr.shape[0] > max_configs:
    indices = np.linspace(0, arr.shape[0] - 1, max_configs, dtype=int)
    arr = arr[indices]
  return np.asarray([
      lattices.send_positions_to_first_unit_cell(config, params.lat_vec, params.rec)
      for config in arr
  ])


def _cell_outline(lat_vec: np.ndarray) -> np.ndarray:
  corners_frac = np.array([
      [-0.5, -0.5],
      [0.5, -0.5],
      [0.5, 0.5],
      [-0.5, 0.5],
      [-0.5, -0.5],
  ])
  return np.einsum("ij,kj->ki", lat_vec, corners_frac)


def _minimum_image(params: RunParams, displacements: np.ndarray) -> np.ndarray:
  frac = np.einsum("ij,...j->...i", np.linalg.inv(params.lat_vec), displacements)
  frac = (frac + 0.5) % 1.0 - 0.5
  return np.einsum("ij,...j->...i", params.lat_vec, frac)


def _draw_density_panel(ax, obs: RunObservables, mode: str) -> None:
  params = obs.params
  if mode == "overall":
    flat = obs.positions.reshape((-1, 2))
    color = "#2f6f9f"
  elif mode == "top":
    flat = obs.positions[:, params.layer_assignment == 1.0, :].reshape((-1, 2))
    color = "#b23a48"
  elif mode == "bottom":
    flat = obs.positions[:, params.layer_assignment == -1.0, :].reshape((-1, 2))
    color = "#2f7d57"
  else:
    raise ValueError(f"Unknown density mode: {mode}")

  ax.scatter(
      flat[:, 0],
      flat[:, 1],
      s=0.55,
      alpha=0.45,
      color=color,
      linewidths=0,
      rasterized=True)

  outline = _cell_outline(params.lat_vec)
  ax.plot(outline[:, 0], outline[:, 1], color="#333333",
          linewidth=0.45, alpha=0.45)
  pad = 0.04 * np.max(np.ptp(outline, axis=0))
  ax.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
  ax.set_ylim(outline[:, 1].min() - pad, outline[:, 1].max() + pad)
  ax.set_aspect("equal", adjustable="box")
  ax.set_xticks([])
  ax.set_yticks([])


def _compute_structure_factor(
    params: RunParams,
    positions: np.ndarray,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray]:
  ms = []
  vals = []
  for m1 in range(-kmax, kmax + 1):
    for m2 in range(-kmax, kmax + 1):
      if m1 == 0 and m2 == 0:
        continue
      kvec = params.rec @ np.array([m1, m2])
      phases = np.exp(1j * np.einsum("cnd,d->cn", positions, kvec))
      rho_k = np.sum(phases, axis=1)
      sk = np.mean(np.abs(rho_k) ** 2) / params.num_bosons
      ms.append((m1, m2))
      vals.append(sk)
  return np.asarray(ms), np.asarray(vals)


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
  ideal_counts = positions.shape[0] * np.count_nonzero(mask_a) * density_b
  ideal_counts = ideal_counts * shell_areas
  if same_layer:
    ideal_counts /= 2.0
  gr = np.divide(
      counts,
      ideal_counts,
      out=np.zeros_like(counts, dtype=float),
      where=ideal_counts > 0)
  return centers, gr


def _load_observables(
    run_dir: Path,
    load_n_ckpts: int,
    max_configs: int | None,
    kmax: int,
    pair_correlation_bins: int,
) -> RunObservables:
  params = _run_params(run_dir)
  positions = _load_positions(params, load_n_ckpts, max_configs)
  ms, vals = _compute_structure_factor(params, positions, kmax)
  top_mask = params.layer_assignment == 1.0
  bottom_mask = params.layer_assignment == -1.0
  pair_r, top_top_gr = _compute_layer_pair_correlation(
      params, positions, top_mask, top_mask, pair_correlation_bins)
  _, bottom_bottom_gr = _compute_layer_pair_correlation(
      params, positions, bottom_mask, bottom_mask, pair_correlation_bins)
  _, top_bottom_gr = _compute_layer_pair_correlation(
      params, positions, top_mask, bottom_mask, pair_correlation_bins)
  return RunObservables(
      params=params,
      positions=positions,
      structure_ms=ms,
      structure_vals=vals,
      pair_r=pair_r,
      top_top_gr=top_top_gr,
      bottom_bottom_gr=bottom_bottom_gr,
      top_bottom_gr=top_bottom_gr,
  )


def _deduplicate_run_dirs(
    run_dirs: list[Path],
    x_param: str,
) -> tuple[list[Path], int]:
  """Keep one run for duplicate scan coordinates using evaluator checkpoint score."""
  selected: dict[tuple[float, float], Path] = {}
  duplicates = 0
  for run_dir in run_dirs:
    parsed = common.parse_run_dir(run_dir)
    x_value = float(parsed["dipole"]) if x_param == "D" else float(parsed["rs"])
    key = (x_value, float(parsed["d"]))
    existing = selected.get(key)
    if existing is None:
      selected[key] = run_dir
      continue
    duplicates += 1
    if common.run_selection_score(run_dir) < common.run_selection_score(existing):
      selected[key] = run_dir
  return sorted(selected.values()), duplicates


def _plot_density_grid(
    observables: list[RunObservables],
    x_values: list[float],
    d_values: list[float],
    output_path: Path,
    mode: str,
    title: str,
    x_param: str,
) -> None:
  x_label = common.x_label(x_param)
  obs_by_param = {
      (common.x_value(obs.params, x_param), obs.params.d): obs
      for obs in observables
  }
  plot_d_vals = common.reversed_d_values(d_values)
  fig_width = max(11.0, 2.10 * len(x_values) + 0.75)
  fig_height = max(7.0, 1.85 * len(plot_d_vals))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_vals),
      len(x_values),
      wspace=0.035,
      hspace=0.055)
  axs = np.empty((len(plot_d_vals), len(x_values)), dtype=object)
  for row in range(len(plot_d_vals)):
    for col in range(len(x_values)):
      axs[row, col] = fig.add_subplot(gridspec[row, col])
  nconfigs = sorted({obs.positions.shape[0] for obs in observables})
  subtitle = (
      f"{nconfigs[0]} configs/run" if len(nconfigs) == 1
      else f"{min(nconfigs)}-{max(nconfigs)} configs/run")
  fig.suptitle(f"{title} ({subtitle})", fontsize=18, y=1.025)
  fig.supylabel("d", fontsize=10)

  for row, d_value in enumerate(plot_d_vals):
    for col, x_value in enumerate(x_values):
      ax = axs[row, col]
      obs = obs_by_param.get((x_value, d_value))
      if obs is None:
        common.draw_missing_panel(ax)
      else:
        _draw_density_panel(ax, obs, mode)
      if row == 0:
        ax.set_title(f"{x_label}={x_value:g}", fontsize=9)
      if col == 0:
        ax.text(
            -0.18,
            0.5,
            f"d={d_value:g}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=0,
            fontsize=7.5)

  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def _plot_structure_grid(
    observables: list[RunObservables],
    x_values: list[float],
    d_values: list[float],
    output_path: Path,
    x_param: str,
) -> None:
  x_label = common.x_label(x_param)
  scan_name = common.scan_name(x_param)
  obs_by_param = {
      (common.x_value(obs.params, x_param), obs.params.d): obs
      for obs in observables
  }
  all_vals = np.concatenate([obs.structure_vals for obs in observables])
  norm = common.finite_percentile_norm(all_vals)
  cmap = plt.get_cmap("viridis")
  plot_d_vals = common.reversed_d_values(d_values)
  fig_width = max(11.0, 2.05 * len(x_values) + 0.75)
  fig_height = max(7.0, 1.80 * len(plot_d_vals))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_vals),
      len(x_values) + 1,
      width_ratios=[1.0] * len(x_values) + [0.055],
      wspace=0.035,
      hspace=0.055)
  axs = np.empty((len(plot_d_vals), len(x_values)), dtype=object)
  for row in range(len(plot_d_vals)):
    for col in range(len(x_values)):
      axs[row, col] = fig.add_subplot(gridspec[row, col])
  cbar_ax = fig.add_subplot(gridspec[:, -1])
  fig.suptitle(
      f"Bilayer boson {scan_name} scan: static structure factor",
      fontsize=18,
      y=1.025)
  fig.supylabel("d", fontsize=10)

  image = None
  for row, d_value in enumerate(plot_d_vals):
    for col, x_value in enumerate(x_values):
      ax = axs[row, col]
      obs = obs_by_param.get((x_value, d_value))
      if obs is None:
        common.draw_missing_panel(ax)
      else:
        sizes = 10 + 18 * np.clip(norm(obs.structure_vals), 0.0, 1.0)
        image = ax.scatter(
            obs.structure_ms[:, 0],
            obs.structure_ms[:, 1],
            c=obs.structure_vals,
            s=sizes,
            cmap=cmap,
            norm=norm,
            linewidths=0)
        ax.axhline(0, color="black", linewidth=0.35, alpha=0.28)
        ax.axvline(0, color="black", linewidth=0.35, alpha=0.28)
        peak_idx = int(np.argmax(obs.structure_vals))
        peak_m = obs.structure_ms[peak_idx]
        peak_value = obs.structure_vals[peak_idx]
        ax.scatter(
            [peak_m[0]], [peak_m[1]],
            marker="o", s=46, facecolors="none", edgecolors="black",
            linewidths=0.8)
        ax.text(
            0.04,
            0.94,
            f"max {peak_value:.2g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            })
        ax.set_aspect("equal", adjustable="box")
      ax.set_xticks([])
      ax.set_yticks([])
      if row == 0:
        ax.set_title(f"{x_label}={x_value:g}", fontsize=9)
      if col == 0:
        ax.text(
            -0.18,
            0.5,
            f"{d_value:g}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=7.5)

  if image is not None:
    cbar = fig.colorbar(image, cax=cbar_ax, extend="max")
    cbar.set_label("S(k)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def _plot_pair_correlation_grid(
    observables: list[RunObservables],
    x_values: list[float],
    d_values: list[float],
    output_path: Path,
    mode: str,
    title: str,
    x_param: str,
) -> None:
  x_label = common.x_label(x_param)
  obs_by_param = {
      (common.x_value(obs.params, x_param), obs.params.d): obs
      for obs in observables
  }
  plot_d_vals = common.reversed_d_values(d_values)
  fig_width = max(11.0, 2.15 * len(x_values) + 0.75)
  fig_height = max(7.0, 1.85 * len(plot_d_vals))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_vals) + 1,
      len(x_values),
      height_ratios=[0.18] + [1.0] * len(plot_d_vals),
      wspace=0.04,
      hspace=0.08)
  legend_ax = fig.add_subplot(gridspec[0, :])
  legend_ax.axis("off")
  axs = np.empty((len(plot_d_vals), len(x_values)), dtype=object)
  for row in range(len(plot_d_vals)):
    for col in range(len(x_values)):
      axs[row, col] = fig.add_subplot(gridspec[row + 1, col])
  fig.suptitle(title, fontsize=18, y=1.025)
  fig.supylabel("d", fontsize=10)
  legend_handles = None
  legend_labels = None

  all_gr = []
  for obs in observables:
    if mode == "same":
      all_gr.extend([obs.top_top_gr, obs.bottom_bottom_gr])
    elif mode == "interlayer":
      all_gr.append(obs.top_bottom_gr)
    else:
      raise ValueError(f"Unknown pair correlation mode: {mode}")
  finite_gr = np.concatenate(all_gr)
  finite_gr = finite_gr[np.isfinite(finite_gr)]
  ymax = float(np.percentile(finite_gr, 98.0)) if finite_gr.size else 1.0
  ymax = max(1.05, 1.10 * ymax)

  for row, d_value in enumerate(plot_d_vals):
    for col, x_value in enumerate(x_values):
      ax = axs[row, col]
      obs = obs_by_param.get((x_value, d_value))
      if obs is None:
        common.draw_missing_panel(ax)
      else:
        ax.axhline(1.0, color="#777777", linewidth=0.55, alpha=0.50)
        if mode == "same":
          ax.plot(
              obs.pair_r,
              obs.top_top_gr,
              color="#b23a48",
              linewidth=1.0,
              label="top-top")
          ax.plot(
              obs.pair_r,
              obs.bottom_bottom_gr,
              color="#2f7d57",
              linewidth=1.0,
              linestyle="--",
              label="bottom-bottom")
        elif mode == "interlayer":
          ax.plot(
              obs.pair_r,
              obs.top_bottom_gr,
              color="#3b5ba9",
              linewidth=1.1,
              label="top-bottom")
        if legend_handles is None:
          legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_xlim(0.0, float(obs.pair_r[-1]))
        ax.set_ylim(0.0, ymax)
        ax.tick_params(labelsize=6, length=2.0, width=0.5)
        if row != len(plot_d_vals) - 1:
          ax.set_xticklabels([])
        if col != 0:
          ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.55)
        ax.spines["bottom"].set_linewidth(0.55)
      if row == 0:
        ax.set_title(f"{x_label}={x_value:g}", fontsize=9)
      if col == 0:
        ax.text(
            -0.22,
            0.5,
            f"d={d_value:g}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=0,
            fontsize=7.5)

  if legend_handles:
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        fontsize=8,
        frameon=False)
  fig.supxlabel("r", fontsize=10)
  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def _plot_structure_summary(
    observables: list[RunObservables],
    x_values: list[float],
    d_values: list[float],
    output_path: Path,
    x_param: str,
) -> None:
  x_label = common.x_label(x_param)
  scan_name = common.scan_name(x_param)
  max_grid = np.full((len(d_values), len(x_values)), np.nan)
  peak_radius_grid = np.full_like(max_grid, np.nan, dtype=float)
  peak_labels = np.full((len(d_values), len(x_values)), "", dtype=object)
  x_index = {value: idx for idx, value in enumerate(x_values)}
  d_index = {value: idx for idx, value in enumerate(d_values)}
  for obs in observables:
    x_idx = x_index[common.x_value(obs.params, x_param)]
    y_idx = d_index[obs.params.d]
    peak_idx = int(np.argmax(obs.structure_vals))
    peak_m = obs.structure_ms[peak_idx]
    max_grid[y_idx, x_idx] = obs.structure_vals[peak_idx]
    peak_radius_grid[y_idx, x_idx] = np.linalg.norm(peak_m)
    peak_labels[y_idx, x_idx] = f"({peak_m[0]},{peak_m[1]})"

  fig, axs = plt.subplots(
      1, 2, figsize=(12, 4.8), squeeze=False, constrained_layout=True)
  fig.suptitle(
      f"Bilayer boson {scan_name} scan: structure factor summary",
      fontsize=16,
      y=1.03)
  panels = [
      (axs[0, 0], max_grid, "max S(k)", "viridis",
       common.finite_percentile_norm(max_grid), None),
      (axs[0, 1], peak_radius_grid, "dominant reciprocal index m", "magma",
       common.finite_norm(peak_radius_grid), peak_labels),
  ]
  for ax, grid, title, cmap_name, norm, labels in panels:
    cmap = plt.get_cmap(cmap_name)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("d")
    ax.set_xticks(
        range(len(x_values)), [f"{value:g}" for value in x_values])
    ax.set_yticks(range(len(d_values)), [f"{value:g}" for value in d_values])
    for y_idx in range(len(d_values)):
      for x_idx in range(len(x_values)):
        value = grid[y_idx, x_idx]
        if np.isfinite(value):
          label = labels[y_idx, x_idx] if labels is not None else f"{value:.2g}"
          ax.text(
              x_idx,
              y_idx,
              label,
              ha="center",
              va="center",
              fontsize=7,
              color=common.text_color_for_value(cmap, norm, value))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = "|m|" if labels is not None else title
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)
  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--x-param",
      choices=("rs", "D"),
      required=True,
      help="Scan coordinate to place on the x axis.",
  )
  parser.add_argument(
      "--run-dir",
      type=Path,
      default=None,
      help="Plot one run directory directly; ignores --scan-dir and --pattern.",
  )
  parser.add_argument("--scan-dir", type=Path, default=None)
  parser.add_argument("--pattern", default=DEFAULT_PATTERN)
  parser.add_argument("--load-n-ckpts", type=int, default=1)
  parser.add_argument(
      "--max-configs",
      type=int,
      default=0,
      help="Subsample configurations per run for plotting and S(k); use 0 for all.",
  )
  parser.add_argument("--kmax", type=int, default=5)
  parser.add_argument("--pair-correlation-bins", type=int, default=80)
  parser.add_argument("--output-prefix", default=None)
  args = parser.parse_args()
  x_param = common.normalize_x_param(args.x_param)
  if args.pair_correlation_bins <= 0:
    raise ValueError("--pair-correlation-bins must be positive")

  scan_dir = (args.scan_dir or common.default_scan_dir(x_param)).resolve()
  max_configs = None if args.max_configs == 0 else args.max_configs
  if args.run_dir is not None:
    run_dirs = [args.run_dir.resolve()]
    scan_dir = run_dirs[0]
  else:
    run_dirs = sorted(path for path in scan_dir.glob(args.pattern) if path.is_dir())
  if not run_dirs:
    raise ValueError(f"No run directories matched {scan_dir / args.pattern}")
  run_dirs, duplicate_count = _deduplicate_run_dirs(run_dirs, x_param)

  observables = []
  skipped = []
  for idx, run_dir in enumerate(run_dirs, start=1):
    try:
      observables.append(
          _load_observables(
              run_dir,
              args.load_n_ckpts,
              max_configs,
              args.kmax,
              args.pair_correlation_bins))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      skipped.append((run_dir, exc))
    print(f"Processed {idx}/{len(run_dirs)}: {run_dir.name}", flush=True)

  if not observables:
    details = "\n".join(f"{path}: {exc}" for path, exc in skipped[:5])
    raise ValueError(f"No usable checkpoints found in {scan_dir}\n{details}")

  x_values = sorted({common.x_value(obs.params, x_param) for obs in observables})
  d_values = sorted({obs.params.d for obs in observables})
  output_prefix = args.output_prefix or common.default_output_prefix(
      "observables", x_param)
  density_output = scan_dir / f"{output_prefix}_density_grid.png"
  top_density_output = scan_dir / f"{output_prefix}_density_top_grid.png"
  bottom_density_output = scan_dir / f"{output_prefix}_density_bottom_grid.png"
  structure_output = scan_dir / f"{output_prefix}_structure_factor_grid.png"
  structure_summary_output = scan_dir / f"{output_prefix}_structure_summary.png"
  same_layer_gr_output = (
      scan_dir / f"{output_prefix}_pair_correlation_same_layer_grid.png")
  top_bottom_gr_output = (
      scan_dir / f"{output_prefix}_pair_correlation_top_bottom_grid.png")
  scan_name = common.scan_name(x_param)
  _plot_density_grid(
      observables,
      x_values,
      d_values,
      density_output,
      mode="overall",
      title=f"Bilayer boson {scan_name} scan: overall xy density",
      x_param=x_param)
  _plot_density_grid(
      observables,
      x_values,
      d_values,
      top_density_output,
      mode="top",
      title=f"Bilayer boson {scan_name} scan: top-layer xy density",
      x_param=x_param)
  _plot_density_grid(
      observables,
      x_values,
      d_values,
      bottom_density_output,
      mode="bottom",
      title=f"Bilayer boson {scan_name} scan: bottom-layer xy density",
      x_param=x_param)
  _plot_structure_grid(observables, x_values, d_values, structure_output, x_param)
  _plot_structure_summary(
      observables,
      x_values,
      d_values,
      structure_summary_output,
      x_param)
  _plot_pair_correlation_grid(
      observables,
      x_values,
      d_values,
      same_layer_gr_output,
      mode="same",
      title=f"Bilayer boson {scan_name} scan: same-layer pair correlation",
      x_param=x_param)
  _plot_pair_correlation_grid(
      observables,
      x_values,
      d_values,
      top_bottom_gr_output,
      mode="interlayer",
      title=f"Bilayer boson {scan_name} scan: top-bottom pair correlation",
      x_param=x_param)

  print(f"Loaded observables for {len(observables)} runs from {scan_dir}")
  if duplicate_count:
    print(
        f"Collapsed {duplicate_count} duplicate "
        f"({common.x_label(x_param)}, d) runs by checkpoint score")
  if skipped:
    print(f"Skipped {len(skipped)} runs")
    for path, exc in skipped[:5]:
      print(f"  {path.name}: {exc}")
  print(f"Saved density grid to {density_output}")
  print(f"Saved top-layer density grid to {top_density_output}")
  print(f"Saved bottom-layer density grid to {bottom_density_output}")
  print(f"Saved structure factor grid to {structure_output}")
  print(f"Saved structure factor summary to {structure_summary_output}")
  print(f"Saved same-layer pair correlation grid to {same_layer_gr_output}")
  print(f"Saved top-bottom pair correlation grid to {top_bottom_gr_output}")


if __name__ == "__main__":
  main()
