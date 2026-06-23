#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot training diagnostics across a bilayer boson scan."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

import plot_scan_common as common


REPO_ROOT = common.REPO_ROOT
DEFAULT_PATTERN = common.DEFAULT_PATTERN


@dataclass(frozen=True)
class RunDiagnostics:
  path: Path
  rs: float
  d: float
  dipole_strength: float
  num_bosons: int
  data: pd.DataFrame
  energy_per_boson: pd.Series
  rolling_energy_per_boson: pd.Series
  final_energy_per_boson: float
  best_rolling_energy_per_boson: float
  final_locstd_per_boson: float | None
  final_pmove: float | None
  final_step: int
  best_checkpoint_step: int
  best_checkpoint_score: float
  best_kinetic_energy: float
  best_potential_intra: float
  best_potential_inter: float
  kinetic_weight: float
  potential_intra_weight: float
  potential_inter_weight: float


def _diagnose_run(
    run_dir: Path,
    burn_in_cut: int,
    rolling_window: int,
) -> RunDiagnostics | None:
  parsed = common.parse_run_dir(run_dir)
  stats = common.load_train_stats(run_dir)
  if stats is None:
    return None
  if "energy" not in stats or "step" not in stats:
    return None
  component_cols = [
      "kinetic_energy",
      "potential_intra",
      "potential_inter",
  ]
  if any(col not in stats for col in component_cols):
    return None
  filtered = stats[stats["step"] >= burn_in_cut].reset_index(drop=True)
  if not filtered.empty:
    stats = filtered

  num_bosons = common.num_bosons(run_dir, parsed)
  energy_per_boson = stats["energy"]
  rolling_rows = common.rolling_rows_from_step_window(stats["step"], rolling_window)
  rolling = energy_per_boson.rolling(
      rolling_rows, min_periods=max(1, rolling_rows // 5)).mean()
  last = stats.iloc[-1]
  final_locstd = None
  if "locstd" in stats:
    final_locstd = float(last["locstd"])
  final_pmove = float(last["pmove"]) if "pmove" in stats else None
  best_step, best_score = common.best_checkpoint_record(run_dir)
  best_row = common.nearest_step_row(stats, best_step)
  best_kinetic = float(best_row["kinetic_energy"])
  best_intra = float(best_row["potential_intra"])
  best_inter = float(best_row["potential_inter"])
  component_norm = abs(best_kinetic) + abs(best_intra) + abs(best_inter)
  if component_norm <= 0.0 or not np.isfinite(component_norm):
    return None

  return RunDiagnostics(
      path=run_dir,
      rs=float(parsed["rs"]),
      d=float(parsed["d"]),
      dipole_strength=float(parsed["dipole"]),
      num_bosons=num_bosons,
      data=stats,
      energy_per_boson=energy_per_boson,
      rolling_energy_per_boson=rolling,
      final_energy_per_boson=float(energy_per_boson.iloc[-1]),
      best_rolling_energy_per_boson=float(rolling.min()),
      final_locstd_per_boson=final_locstd,
      final_pmove=final_pmove,
      final_step=int(last["step"]),
      best_checkpoint_step=int(best_row["step"]),
      best_checkpoint_score=best_score,
      best_kinetic_energy=best_kinetic,
      best_potential_intra=best_intra,
      best_potential_inter=best_inter,
      kinetic_weight=abs(best_kinetic) / component_norm,
      potential_intra_weight=abs(best_intra) / component_norm,
      potential_inter_weight=abs(best_inter) / component_norm,
  )


def _deduplicate_runs(
    runs: list[RunDiagnostics],
    x_param: str,
) -> tuple[list[RunDiagnostics], int]:
  """Keep the lowest best rolling energy for duplicate scan coordinates."""
  best_by_param: dict[tuple[float, float], RunDiagnostics] = {}
  duplicates = 0
  for run in runs:
    key = (common.x_value(run, x_param), run.d)
    existing = best_by_param.get(key)
    if existing is None:
      best_by_param[key] = run
      continue
    duplicates += 1
    if run.best_checkpoint_score < existing.best_checkpoint_score:
      best_by_param[key] = run
  return sorted(
      best_by_param.values(), key=lambda run: (run.d, common.x_value(run, x_param))
  ), duplicates


def _metric_grid(
    runs: list[RunDiagnostics],
    x_values: list[float],
    d_values: list[float],
    attr: str,
    x_param: str,
) -> np.ndarray:
  grid = np.full((len(d_values), len(x_values)), np.nan)
  x_index = {value: idx for idx, value in enumerate(x_values)}
  d_index = {value: idx for idx, value in enumerate(d_values)}
  for run in runs:
    value = getattr(run, attr)
    if value is not None:
      grid[d_index[run.d], x_index[common.x_value(run, x_param)]] = value
  return grid


def _format_metric_value(value: float, attr: str) -> str:
  if attr == "final_pmove":
    return f"{value:.2f}"
  return f"{value:.3g}"


def _plot_scan_energy_grid(
    runs: list[RunDiagnostics],
    x_values: list[float],
    d_values: list[float],
    output_path: Path,
    x_param: str,
) -> None:
  x_label = common.x_label(x_param)
  scan_name = common.scan_name(x_param)
  run_by_param = {(common.x_value(run, x_param), run.d): run for run in runs}
  final_grid = _metric_grid(
      runs, x_values, d_values, "final_energy_per_boson", x_param)
  cmap = plt.get_cmap("viridis")
  norm = common.finite_norm(final_grid)

  fig_width = max(11.0, 2.35 * len(x_values) + 0.75)
  plot_d_values = list(reversed(d_values))
  fig_height = max(7.0, 1.75 * len(plot_d_values))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_values),
      len(x_values) + 1,
      width_ratios=[1.0] * len(x_values) + [0.055],
      wspace=0.035,
      hspace=0.055)
  axs = np.empty((len(plot_d_values), len(x_values)), dtype=object)
  for row in range(len(plot_d_values)):
    for col in range(len(x_values)):
      sharex = axs[0, col] if row > 0 else None
      axs[row, col] = fig.add_subplot(gridspec[row, col], sharex=sharex)
  cbar_ax = fig.add_subplot(gridspec[:, -1])
  fig.suptitle(
      f"Bilayer boson {scan_name} scan diagnostics: energy / N convergence",
      fontsize=18,
      y=1.025)
  fig.supylabel("energy / N", fontsize=10)

  for row, d_value in enumerate(plot_d_values):
    for col, x_value in enumerate(x_values):
      ax = axs[row, col]
      run = run_by_param.get((x_value, d_value))
      ax.grid(alpha=0.20, linewidth=0.4)
      ax.tick_params(labelsize=7, length=2)
      if run is None:
        common.draw_missing_panel(ax)
      else:
        color = cmap(norm(run.final_energy_per_boson))
        steps = run.data["step"]
        ax.plot(
            steps,
            run.energy_per_boson,
            color=color,
            linewidth=0.35,
            alpha=0.22)
        ax.plot(
            steps,
            run.rolling_energy_per_boson,
            color=color,
            linewidth=1.15)
        if "ewmean" in run.data:
          ax.plot(
              steps,
              run.data["ewmean"],
              color="#111111",
              linewidth=0.7,
              alpha=0.55)
        label = f"E/N {run.final_energy_per_boson:.3g}\n"
        label += f"std/N {run.final_locstd_per_boson:.2g}" if (
            run.final_locstd_per_boson is not None) else "std/N n/a"
        if run.final_pmove is not None:
          label += f"\npmove {run.final_pmove:.2f}"
        ax.text(
            0.03,
            0.96,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            })

      if row == 0:
        ax.set_title(f"{x_label}={x_value:g}", fontsize=9)
      if col == 0:
        ax.text(
            -0.30,
            0.5,
            f"d={d_value:g}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=7.5)
      else:
        ax.tick_params(labelleft=False)
      if row == len(plot_d_values) - 1:
        ax.set_xlabel("step", fontsize=8)
      else:
        ax.tick_params(labelbottom=False)

  sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, cax=cbar_ax)
  cbar.set_label("final energy / N", fontsize=9)
  cbar.ax.tick_params(labelsize=8)
  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def _plot_energy_component_bars(
    runs: list[RunDiagnostics],
    x_values: list[float],
    d_values: list[float],
    output_path: Path,
    x_param: str,
) -> None:
  x_label = common.x_label(x_param)
  run_by_param = {(common.x_value(run, x_param), run.d): run for run in runs}
  plot_d_values = list(reversed(d_values))
  fig_width = max(11.0, 2.15 * len(x_values) + 0.75)
  fig_height = max(7.0, 1.55 * len(plot_d_values))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_values) + 1,
      len(x_values),
      height_ratios=[0.18] + [1.0] * len(plot_d_values),
      wspace=0.035,
      hspace=0.055)
  legend_ax = fig.add_subplot(gridspec[0, :])
  legend_ax.axis("off")
  axs = np.empty((len(plot_d_values), len(x_values)), dtype=object)
  for row in range(len(plot_d_values)):
    for col in range(len(x_values)):
      axs[row, col] = fig.add_subplot(gridspec[row + 1, col])

  fig.suptitle("Best-checkpoint energy component weights", fontsize=18, y=1.025)
  fig.supylabel("d", fontsize=10)
  components = [
      ("|K|", "kinetic_weight", "best_kinetic_energy", "#4c78a8"),
      ("|V_intra|", "potential_intra_weight", "best_potential_intra", "#59a14f"),
      ("|V_inter|", "potential_inter_weight", "best_potential_inter", "#e15759"),
  ]
  legend_handles = []
  legend_labels = []

  for row, d_value in enumerate(plot_d_values):
    for col, x_value in enumerate(x_values):
      ax = axs[row, col]
      run = run_by_param.get((x_value, d_value))
      if run is None:
        common.draw_missing_panel(ax)
      else:
        left = 0.0
        for label, weight_attr, energy_attr, color in components:
          width = getattr(run, weight_attr)
          bars = ax.barh(
              [0],
              [width],
              left=[left],
              height=0.46,
              color=color,
              edgecolor="white",
              linewidth=0.55,
              label=label)
          if width >= 0.12:
            signed_energy = getattr(run, energy_attr)
            ax.text(
                left + width / 2.0,
                0,
                f"{width:.2f}\n{signed_energy:.2g}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="white")
          left += width
          if len(legend_handles) < len(components):
            legend_handles.append(bars[0])
            legend_labels.append(label)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=6.5, length=2)
        ax.grid(axis="x", alpha=0.18, linewidth=0.45)
        ax.text(
            0.03,
            0.90,
            f"step {run.best_checkpoint_step}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color="#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.55)
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
      if row != len(plot_d_values) - 1:
        ax.tick_params(labelbottom=False)

  if legend_handles:
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=3,
        fontsize=8,
        frameon=False)
  fig.supxlabel("fraction of |K| + |V_intra| + |V_inter|", fontsize=10)

  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def _plot_scan_metric_summary(
    runs: list[RunDiagnostics],
    x_values: list[float],
    d_values: list[float],
    output_path: Path,
    x_param: str,
) -> None:
  x_label = common.x_label(x_param)
  scan_name = common.scan_name(x_param)
  metrics = [
      ("final_energy_per_boson", "final energy / N", "RdBu_r", "centered"),
      ("best_rolling_energy_per_boson", "best rolling energy / N", "RdBu_r",
       "centered"),
      ("final_locstd_per_boson", "final std(E_L) / N", "magma_r",
       "percentile"),
      ("final_pmove", "final pmove", "cividis", "pmove"),
  ]
  fig, axs = plt.subplots(
      2, 2, figsize=(12, 9), squeeze=False, constrained_layout=True)
  fig.suptitle(f"Bilayer boson {scan_name} scan summary", fontsize=16, y=1.02)

  for ax, (attr, title, cmap_name, norm_kind) in zip(axs.ravel(), metrics):
    grid = _metric_grid(runs, x_values, d_values, attr, x_param)
    if norm_kind == "centered":
      norm = common.finite_centered_norm(grid)
    elif norm_kind == "pmove":
      norm = Normalize(vmin=0.0, vmax=1.0)
    else:
      norm = common.finite_percentile_norm(grid)
    cmap = plt.get_cmap(cmap_name)
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm)
    ax.set_title(title)
    ax.set_xticks(
        range(len(x_values)), [f"{value:g}" for value in x_values])
    ax.set_yticks(range(len(d_values)), [f"{value:g}" for value in d_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel("d")
    for y_idx in range(len(d_values)):
      for x_idx in range(len(x_values)):
        value = grid[y_idx, x_idx]
        if np.isfinite(value):
          ax.text(
              x_idx,
              y_idx,
              _format_metric_value(value, attr),
              ha="center",
              va="center",
              fontsize=7,
              color=common.text_color_for_value(cmap, norm, value))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title, fontsize=8)
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
  parser.add_argument("--burn-in-cut", type=int, default=0)
  parser.add_argument(
      "--rolling-window",
      type=int,
      default=100,
      help="Rolling-mean window in optimizer steps, not saved CSV rows.",
  )
  parser.add_argument("--output-prefix", default=None)
  args = parser.parse_args()
  x_param = common.normalize_x_param(args.x_param)

  scan_dir = (args.scan_dir or common.default_scan_dir(x_param)).resolve()
  if args.run_dir is not None:
    run_dirs = [args.run_dir.resolve()]
    scan_dir = run_dirs[0]
  else:
    run_dirs = sorted(path for path in scan_dir.glob(args.pattern) if path.is_dir())
  if not run_dirs:
    raise ValueError(f"No run directories matched {scan_dir / args.pattern}")

  runs = []
  skipped = []
  for run_dir in run_dirs:
    diagnostics = _diagnose_run(run_dir, args.burn_in_cut, args.rolling_window)
    if diagnostics is None:
      skipped.append(run_dir)
    else:
      runs.append(diagnostics)
  if not runs:
    raise ValueError(f"No usable train_stats.csv files found in {scan_dir}")
  runs, duplicate_count = _deduplicate_runs(runs, x_param)

  x_values = sorted({common.x_value(run, x_param) for run in runs})
  d_values = sorted({run.d for run in runs})
  output_prefix = args.output_prefix or common.default_output_prefix(
      "diagnostics", x_param)
  component_bars_output = (
      scan_dir / f"{output_prefix}_energy_component_bars.png")
  summary_output = scan_dir / f"{output_prefix}_summary.png"
  _plot_energy_component_bars(
      runs, x_values, d_values, component_bars_output, x_param)
  _plot_scan_metric_summary(runs, x_values, d_values, summary_output, x_param)

  print(f"Loaded {len(runs)} runs from {scan_dir}")
  if duplicate_count:
    print(
        f"Collapsed {duplicate_count} duplicate "
        f"({common.x_label(x_param)}, d) runs by best checkpoint score")
  if skipped:
    print(
        f"Skipped {len(skipped)} runs without usable training stats/components")
  print(f"Saved energy component bars to {component_bars_output}")
  print(f"Saved summary heatmaps to {summary_output}")


if __name__ == "__main__":
  main()
