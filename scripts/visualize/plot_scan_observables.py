#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot density and structure factor across an rs,d bilayer boson scan."""

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
from matplotlib.colors import Normalize
import numpy as np

from periodicwave.pbc import lattices


DEFAULT_SCAN_DIR = (
    REPO_ROOT
    / "results"
    / "scan_260603"
)
DEFAULT_PATTERN = "N24_layers12_12_rs*_d*_D20.0_sq"
RUN_RE = re.compile(
    r"N(?P<num_bosons>\d+)_layers(?P<layer_a>\d+)_(?P<layer_b>\d+)"
    r"_rs(?P<rs>[0-9.]+)_d(?P<d>[0-9.]+)_D(?P<dipole>[0-9.]+)_"
    r"(?P<cell>[^/]+)$"
)


@dataclass(frozen=True)
class RunParams:
  path: Path
  num_bosons: int
  layer_occupations: tuple[int, int]
  rs: float
  d: float
  dipole_strength: float
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


def _parse_run_dir(run_dir: Path) -> dict[str, str]:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  return match.groupdict()


def _lattice(
    num_bosons: int,
    density_rs: float,
    supercell_shape: str,
) -> np.ndarray:
  if supercell_shape == "tri":
    supercell_a = density_rs * np.sqrt(2 * np.pi / np.sqrt(3) * num_bosons)
    lat_vec, _ = lattices._triangular_lattice_vecs_periodic_potential(
        supercell_a, 1)
    return lat_vec
  if supercell_shape == "sq":
    supercell_a = density_rs * np.sqrt(np.pi * num_bosons)
    return lattices._square_lattice_vecs(supercell_a)
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")


def _run_params(run_dir: Path) -> RunParams:
  parsed = _parse_run_dir(run_dir)
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
      supercell_shape=supercell_shape,
      lat_vec=lat_vec,
      rec=rec,
      layer_assignment=layer_assignment,
  )


def _latest_checkpoint_files(run_dir: Path, nfiles: int) -> list[Path]:
  numbered = []
  for path in run_dir.glob("qmcjax_ckpt_*.npz"):
    try:
      numbered.append((int(path.stem.split("_")[-1]), path))
    except ValueError:
      pass
  return [path for _, path in sorted(numbered, reverse=True)[:nfiles]]


def _load_positions(
    params: RunParams,
    load_n_ckpts: int,
    max_configs: int | None,
) -> np.ndarray:
  _install_jax_array_unpickle_fallback()
  positions = []
  ckpt_files = _latest_checkpoint_files(params.path, load_n_ckpts)
  if not ckpt_files:
    raise ValueError(f"No checkpoints found in {params.path}")
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


def _load_observables(
    run_dir: Path,
    load_n_ckpts: int,
    max_configs: int | None,
    kmax: int,
) -> RunObservables:
  params = _run_params(run_dir)
  positions = _load_positions(params, load_n_ckpts, max_configs)
  ms, vals = _compute_structure_factor(params, positions, kmax)
  return RunObservables(
      params=params,
      positions=positions,
      structure_ms=ms,
      structure_vals=vals,
  )


def _finite_norm(values: np.ndarray) -> Normalize:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return Normalize(vmin=0.0, vmax=1.0)
  vmin = float(finite.min())
  vmax = float(finite.max())
  if np.isclose(vmin, vmax):
    return Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
  return Normalize(vmin=vmin, vmax=vmax)


def _finite_percentile_norm(
    values: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> Normalize:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return Normalize(vmin=0.0, vmax=1.0)
  vmin, vmax = np.percentile(finite, [low, high])
  if np.isclose(vmin, vmax):
    return Normalize(vmin=float(vmin) - 1.0, vmax=float(vmax) + 1.0)
  return Normalize(vmin=float(vmin), vmax=float(vmax))


def _plot_d_values(d_values: list[float]) -> list[float]:
  return list(reversed(d_values))


def _text_color_for_value(
    cmap: matplotlib.colors.Colormap,
    norm: Normalize,
    value: float,
) -> str:
  normalized = norm(value)
  if np.ma.is_masked(normalized):
    return "black"
  red, green, blue, _ = cmap(float(normalized))
  luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
  return "black" if luminance > 0.58 else "white"


def _plot_density_grid(
    observables: list[RunObservables],
    rs_values: list[float],
    d_values: list[float],
    output_path: Path,
    mode: str,
    title: str,
) -> None:
  obs_by_param = {(obs.params.rs, obs.params.d): obs for obs in observables}
  plot_d_vals = _plot_d_values(d_values)
  fig_width = max(11.0, 2.10 * len(rs_values) + 0.75)
  fig_height = max(7.0, 1.85 * len(plot_d_vals))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_vals),
      len(rs_values),
      wspace=0.035,
      hspace=0.055)
  axs = np.empty((len(plot_d_vals), len(rs_values)), dtype=object)
  for row in range(len(plot_d_vals)):
    for col in range(len(rs_values)):
      axs[row, col] = fig.add_subplot(gridspec[row, col])
  nconfigs = sorted({obs.positions.shape[0] for obs in observables})
  subtitle = (
      f"{nconfigs[0]} configs/run" if len(nconfigs) == 1
      else f"{min(nconfigs)}-{max(nconfigs)} configs/run")
  fig.suptitle(f"{title} ({subtitle})", fontsize=18, y=1.025)
  fig.supylabel("d", fontsize=10)

  for row, d_value in enumerate(plot_d_vals):
    for col, rs_value in enumerate(rs_values):
      ax = axs[row, col]
      obs = obs_by_param.get((rs_value, d_value))
      if obs is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
      else:
        _draw_density_panel(ax, obs, mode)
      if row == 0:
        ax.set_title(f"rs={rs_value:g}", fontsize=9)
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
    rs_values: list[float],
    d_values: list[float],
    output_path: Path,
) -> None:
  obs_by_param = {(obs.params.rs, obs.params.d): obs for obs in observables}
  all_vals = np.concatenate([obs.structure_vals for obs in observables])
  norm = _finite_percentile_norm(all_vals)
  cmap = plt.get_cmap("viridis")
  plot_d_vals = _plot_d_values(d_values)
  fig_width = max(11.0, 2.05 * len(rs_values) + 0.75)
  fig_height = max(7.0, 1.80 * len(plot_d_vals))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_vals),
      len(rs_values) + 1,
      width_ratios=[1.0] * len(rs_values) + [0.055],
      wspace=0.035,
      hspace=0.055)
  axs = np.empty((len(plot_d_vals), len(rs_values)), dtype=object)
  for row in range(len(plot_d_vals)):
    for col in range(len(rs_values)):
      axs[row, col] = fig.add_subplot(gridspec[row, col])
  cbar_ax = fig.add_subplot(gridspec[:, -1])
  fig.suptitle("Bilayer boson scan: static structure factor", fontsize=18, y=1.025)
  fig.supylabel("d", fontsize=10)

  image = None
  for row, d_value in enumerate(plot_d_vals):
    for col, rs_value in enumerate(rs_values):
      ax = axs[row, col]
      obs = obs_by_param.get((rs_value, d_value))
      if obs is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
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
        ax.set_title(f"rs={rs_value:g}", fontsize=9)
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


def _plot_structure_summary(
    observables: list[RunObservables],
    rs_values: list[float],
    d_values: list[float],
    output_path: Path,
) -> None:
  max_grid = np.full((len(d_values), len(rs_values)), np.nan)
  peak_radius_grid = np.full_like(max_grid, np.nan, dtype=float)
  peak_labels = np.full((len(d_values), len(rs_values)), "", dtype=object)
  rs_index = {value: idx for idx, value in enumerate(rs_values)}
  d_index = {value: idx for idx, value in enumerate(d_values)}
  for obs in observables:
    x_idx = rs_index[obs.params.rs]
    y_idx = d_index[obs.params.d]
    peak_idx = int(np.argmax(obs.structure_vals))
    peak_m = obs.structure_ms[peak_idx]
    max_grid[y_idx, x_idx] = obs.structure_vals[peak_idx]
    peak_radius_grid[y_idx, x_idx] = np.linalg.norm(peak_m)
    peak_labels[y_idx, x_idx] = f"({peak_m[0]},{peak_m[1]})"

  fig, axs = plt.subplots(
      1, 2, figsize=(12, 4.8), squeeze=False, constrained_layout=True)
  fig.suptitle("Bilayer boson scan: structure factor summary", fontsize=16, y=1.03)
  panels = [
      (axs[0, 0], max_grid, "max S(k)", "viridis",
       _finite_percentile_norm(max_grid), None),
      (axs[0, 1], peak_radius_grid, "dominant reciprocal index m", "magma",
       _finite_norm(peak_radius_grid), peak_labels),
  ]
  for ax, grid, title, cmap_name, norm, labels in panels:
    cmap = plt.get_cmap(cmap_name)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("rs")
    ax.set_ylabel("d")
    ax.set_xticks(range(len(rs_values)), [f"{value:g}" for value in rs_values])
    ax.set_yticks(range(len(d_values)), [f"{value:g}" for value in d_values])
    for y_idx in range(len(d_values)):
      for x_idx in range(len(rs_values)):
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
              color=_text_color_for_value(cmap, norm, value))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = "|m|" if labels is not None else title
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)
  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
  parser.add_argument("--pattern", default=DEFAULT_PATTERN)
  parser.add_argument("--load-n-ckpts", type=int, default=1)
  parser.add_argument(
      "--max-configs",
      type=int,
      default=0,
      help="Subsample configurations per run for plotting and S(k); use 0 for all.",
  )
  parser.add_argument("--kmax", type=int, default=5)
  parser.add_argument("--output-prefix", default="scan_observables")
  args = parser.parse_args()

  scan_dir = args.scan_dir.resolve()
  max_configs = None if args.max_configs == 0 else args.max_configs
  run_dirs = sorted(path for path in scan_dir.glob(args.pattern) if path.is_dir())
  if not run_dirs:
    raise ValueError(f"No run directories matched {scan_dir / args.pattern}")

  observables = []
  skipped = []
  for idx, run_dir in enumerate(run_dirs, start=1):
    try:
      observables.append(
          _load_observables(run_dir, args.load_n_ckpts, max_configs, args.kmax))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      skipped.append((run_dir, exc))
    print(f"Processed {idx}/{len(run_dirs)}: {run_dir.name}", flush=True)

  if not observables:
    details = "\n".join(f"{path}: {exc}" for path, exc in skipped[:5])
    raise ValueError(f"No usable checkpoints found in {scan_dir}\n{details}")

  rs_values = sorted({obs.params.rs for obs in observables})
  d_values = sorted({obs.params.d for obs in observables})
  density_output = scan_dir / f"{args.output_prefix}_density_grid.png"
  top_density_output = scan_dir / f"{args.output_prefix}_density_top_grid.png"
  bottom_density_output = scan_dir / f"{args.output_prefix}_density_bottom_grid.png"
  structure_output = scan_dir / f"{args.output_prefix}_structure_factor_grid.png"
  structure_summary_output = scan_dir / f"{args.output_prefix}_structure_summary.png"
  _plot_density_grid(
      observables,
      rs_values,
      d_values,
      density_output,
      mode="overall",
      title="Bilayer boson scan: overall xy density")
  _plot_density_grid(
      observables,
      rs_values,
      d_values,
      top_density_output,
      mode="top",
      title="Bilayer boson scan: top-layer xy density")
  _plot_density_grid(
      observables,
      rs_values,
      d_values,
      bottom_density_output,
      mode="bottom",
      title="Bilayer boson scan: bottom-layer xy density")
  _plot_structure_grid(observables, rs_values, d_values, structure_output)
  _plot_structure_summary(
      observables,
      rs_values,
      d_values,
      structure_summary_output)

  print(f"Loaded observables for {len(observables)} runs from {scan_dir}")
  if skipped:
    print(f"Skipped {len(skipped)} runs")
    for path, exc in skipped[:5]:
      print(f"  {path.name}: {exc}")
  print(f"Saved density grid to {density_output}")
  print(f"Saved top-layer density grid to {top_density_output}")
  print(f"Saved bottom-layer density grid to {bottom_density_output}")
  print(f"Saved structure factor grid to {structure_output}")
  print(f"Saved structure factor summary to {structure_summary_output}")


if __name__ == "__main__":
  main()
