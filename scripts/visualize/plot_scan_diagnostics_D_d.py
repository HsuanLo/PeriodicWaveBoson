#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot training diagnostics across a D,d bilayer boson scan."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_DIR = (
    REPO_ROOT
    / "results"
    / "D_d_scan_N24_rs1.0_sq"
)
DEFAULT_PATTERN = "N*_layers*_*_rs*_d*_D*_sq"
RUN_RE = re.compile(
    r"N(?P<num_bosons>\d+)_layers(?P<layer_a>\d+)_(?P<layer_b>\d+)"
    r"_rs(?P<rs>[0-9.]+)_d(?P<d>[0-9.]+)_D(?P<dipole>[0-9.]+)"
    r"(?:_seed(?P<seed>\d+))?_"
    r"(?P<cell>sq|tri)(?:_[^/]+)?$"
)


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


def _parse_run_dir(run_dir: Path) -> dict[str, str]:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  return match.groupdict()


def _read_config(run_dir: Path) -> dict:
  config_path = run_dir / "config.json"
  if not config_path.exists():
    return {}
  with config_path.open(encoding="utf-8") as f:
    return json.load(f)


def _num_bosons(run_dir: Path, parsed: dict[str, str]) -> int:
  config = _read_config(run_dir)
  config_bosons = config.get("system", {}).get("bosons", [None])[0]
  if config_bosons:
    return int(config_bosons)
  return int(parsed["num_bosons"])


def _load_train_stats(run_dir: Path) -> pd.DataFrame | None:
  stats_files = [run_dir / "train_stats.csv"] + sorted(
      run_dir.glob("train_stats_*.csv"))
  frames = []
  for stats_file in stats_files:
    if stats_file.exists() and stats_file.stat().st_size > 0:
      frames.append(pd.read_csv(stats_file))
  if not frames:
    return None
  stats = pd.concat(frames, ignore_index=True)
  if stats.empty:
    return None
  return stats.sort_values("step").reset_index(drop=True)


def _best_checkpoint_record(run_dir: Path) -> tuple[int | None, float]:
  manifest_path = run_dir / "qmcjax_best_checkpoints.csv"
  if not manifest_path.exists() or manifest_path.stat().st_size == 0:
    return None, float("inf")
  rows = pd.read_csv(manifest_path)
  if rows.empty or "step" not in rows or "score" not in rows:
    return None, float("inf")
  rows = rows[np.isfinite(rows["score"])]
  if rows.empty:
    return None, float("inf")
  row = rows.loc[rows["score"].idxmin()]
  return int(row["step"]), float(row["score"])


def _nearest_step_row(stats: pd.DataFrame, step: int | None) -> pd.Series:
  if step is None:
    return stats.iloc[-1]
  idx = (stats["step"] - step).abs().idxmin()
  return stats.loc[idx]


def _rolling_rows_from_step_window(steps: pd.Series, rolling_window: int) -> int:
  """Converts a requested optimizer-step window to saved CSV rows."""
  if rolling_window <= 0 or len(steps) < 2:
    return max(1, rolling_window)
  step_diffs = steps.sort_values().diff().dropna()
  step_diffs = step_diffs[step_diffs > 0]
  if step_diffs.empty:
    return max(1, rolling_window)
  step_interval = float(step_diffs.median())
  return max(1, int(round(float(rolling_window) / step_interval)))


def _diagnose_run(
    run_dir: Path,
    burn_in_cut: int,
    rolling_window: int,
) -> RunDiagnostics | None:
  parsed = _parse_run_dir(run_dir)
  stats = _load_train_stats(run_dir)
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

  num_bosons = _num_bosons(run_dir, parsed)
  energy_per_boson = stats["energy"]
  rolling_rows = _rolling_rows_from_step_window(stats["step"], rolling_window)
  rolling = energy_per_boson.rolling(
      rolling_rows, min_periods=max(1, rolling_rows // 5)).mean()
  last = stats.iloc[-1]
  final_locstd = None
  if "locstd" in stats:
    final_locstd = float(last["locstd"])
  final_pmove = float(last["pmove"]) if "pmove" in stats else None
  best_step, best_score = _best_checkpoint_record(run_dir)
  best_row = _nearest_step_row(stats, best_step)
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


def _deduplicate_runs(runs: list[RunDiagnostics]) -> tuple[list[RunDiagnostics], int]:
  """Keep the lowest best rolling energy for duplicate scan coordinates."""
  best_by_param: dict[tuple[float, float], RunDiagnostics] = {}
  duplicates = 0
  for run in runs:
    key = (run.dipole_strength, run.d)
    existing = best_by_param.get(key)
    if existing is None:
      best_by_param[key] = run
      continue
    duplicates += 1
    if run.best_checkpoint_score < existing.best_checkpoint_score:
      best_by_param[key] = run
  return sorted(
      best_by_param.values(), key=lambda run: (run.d, run.dipole_strength)
  ), duplicates


def _metric_grid(
    runs: list[RunDiagnostics],
    dipole_values: list[float],
    d_values: list[float],
    attr: str,
) -> np.ndarray:
  grid = np.full((len(d_values), len(dipole_values)), np.nan)
  dipole_index = {value: idx for idx, value in enumerate(dipole_values)}
  d_index = {value: idx for idx, value in enumerate(d_values)}
  for run in runs:
    value = getattr(run, attr)
    if value is not None:
      grid[d_index[run.d], dipole_index[run.dipole_strength]] = value
  return grid


def _finite_norm(values: np.ndarray) -> Normalize:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return Normalize(vmin=0.0, vmax=1.0)
  if math.isclose(float(finite.min()), float(finite.max())):
    return Normalize(vmin=float(finite.min()) - 1.0, vmax=float(finite.max()) + 1.0)
  return Normalize(vmin=float(finite.min()), vmax=float(finite.max()))


def _finite_percentile_norm(
    values: np.ndarray,
    low: float = 5.0,
    high: float = 95.0,
) -> Normalize:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return Normalize(vmin=0.0, vmax=1.0)
  vmin, vmax = np.percentile(finite, [low, high])
  if math.isclose(float(vmin), float(vmax)):
    return Normalize(vmin=float(vmin) - 1.0, vmax=float(vmax) + 1.0)
  return Normalize(vmin=float(vmin), vmax=float(vmax))


def _finite_centered_norm(values: np.ndarray, center: float = 0.0) -> Normalize:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return TwoSlopeNorm(vmin=-1.0, vcenter=center, vmax=1.0)
  vmin, vmax = np.percentile(finite, [5.0, 95.0])
  vmin = min(float(vmin), center - 1e-6)
  vmax = max(float(vmax), center + 1e-6)
  return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def _annotation_color(cmap: matplotlib.colors.Colormap,
                      norm: Normalize,
                      value: float) -> str:
  normalized = norm(value)
  if np.ma.is_masked(normalized):
    return "black"
  red, green, blue, _ = cmap(float(normalized))
  luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
  return "black" if luminance > 0.58 else "white"


def _format_metric_value(value: float, attr: str) -> str:
  if attr == "final_pmove":
    return f"{value:.2f}"
  return f"{value:.3g}"


def _draw_missing_panel(ax) -> None:
  ax.set_facecolor("#f7f7f5")
  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_color("#d7d7d2")
    spine.set_linewidth(0.6)
  ax.text(
      0.5,
      0.5,
      "not run",
      transform=ax.transAxes,
      ha="center",
      va="center",
      fontsize=7.5,
      color="#777770")


def _plot_scan_energy_grid(
    runs: list[RunDiagnostics],
    dipole_values: list[float],
    d_values: list[float],
    output_path: Path,
) -> None:
  run_by_param = {(run.dipole_strength, run.d): run for run in runs}
  final_grid = _metric_grid(
      runs, dipole_values, d_values, "final_energy_per_boson")
  cmap = plt.get_cmap("viridis")
  norm = _finite_norm(final_grid)

  fig_width = max(11.0, 2.35 * len(dipole_values) + 0.75)
  plot_d_values = list(reversed(d_values))
  fig_height = max(7.0, 1.75 * len(plot_d_values))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_values),
      len(dipole_values) + 1,
      width_ratios=[1.0] * len(dipole_values) + [0.055],
      wspace=0.035,
      hspace=0.055)
  axs = np.empty((len(plot_d_values), len(dipole_values)), dtype=object)
  for row in range(len(plot_d_values)):
    for col in range(len(dipole_values)):
      sharex = axs[0, col] if row > 0 else None
      axs[row, col] = fig.add_subplot(gridspec[row, col], sharex=sharex)
  cbar_ax = fig.add_subplot(gridspec[:, -1])
  fig.suptitle(
      "Bilayer boson D,d scan diagnostics: energy / N convergence",
      fontsize=18,
      y=1.025)
  fig.supylabel("energy / N", fontsize=10)

  for row, d_value in enumerate(plot_d_values):
    for col, dipole_value in enumerate(dipole_values):
      ax = axs[row, col]
      run = run_by_param.get((dipole_value, d_value))
      ax.grid(alpha=0.20, linewidth=0.4)
      ax.tick_params(labelsize=7, length=2)
      if run is None:
        _draw_missing_panel(ax)
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
        ax.set_title(f"D={dipole_value:g}", fontsize=9)
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
    dipole_values: list[float],
    d_values: list[float],
    output_path: Path,
) -> None:
  run_by_param = {(run.dipole_strength, run.d): run for run in runs}
  plot_d_values = list(reversed(d_values))
  fig_width = max(11.0, 2.15 * len(dipole_values) + 0.75)
  fig_height = max(7.0, 1.55 * len(plot_d_values))
  fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
  gridspec = fig.add_gridspec(
      len(plot_d_values) + 1,
      len(dipole_values),
      height_ratios=[0.18] + [1.0] * len(plot_d_values),
      wspace=0.035,
      hspace=0.055)
  legend_ax = fig.add_subplot(gridspec[0, :])
  legend_ax.axis("off")
  axs = np.empty((len(plot_d_values), len(dipole_values)), dtype=object)
  for row in range(len(plot_d_values)):
    for col in range(len(dipole_values)):
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
    for col, dipole_value in enumerate(dipole_values):
      ax = axs[row, col]
      run = run_by_param.get((dipole_value, d_value))
      if run is None:
        _draw_missing_panel(ax)
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
        ax.set_title(f"D={dipole_value:g}", fontsize=9)
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
    dipole_values: list[float],
    d_values: list[float],
    output_path: Path,
) -> None:
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
  fig.suptitle("Bilayer boson D,d scan summary", fontsize=16, y=1.02)

  for ax, (attr, title, cmap_name, norm_kind) in zip(axs.ravel(), metrics):
    grid = _metric_grid(runs, dipole_values, d_values, attr)
    if norm_kind == "centered":
      norm = _finite_centered_norm(grid)
    elif norm_kind == "pmove":
      norm = Normalize(vmin=0.0, vmax=1.0)
    else:
      norm = _finite_percentile_norm(grid)
    cmap = plt.get_cmap(cmap_name)
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm)
    ax.set_title(title)
    ax.set_xticks(
        range(len(dipole_values)), [f"{value:g}" for value in dipole_values])
    ax.set_yticks(range(len(d_values)), [f"{value:g}" for value in d_values])
    ax.set_xlabel("D")
    ax.set_ylabel("d")
    for y_idx in range(len(d_values)):
      for x_idx in range(len(dipole_values)):
        value = grid[y_idx, x_idx]
        if np.isfinite(value):
          ax.text(
              x_idx,
              y_idx,
              _format_metric_value(value, attr),
              ha="center",
              va="center",
              fontsize=7,
              color=_annotation_color(cmap, norm, value))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

  fig.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--run-dir",
      type=Path,
      default=None,
      help="Plot one run directory directly; ignores --scan-dir and --pattern.",
  )
  parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
  parser.add_argument("--pattern", default=DEFAULT_PATTERN)
  parser.add_argument("--burn-in-cut", type=int, default=0)
  parser.add_argument(
      "--rolling-window",
      type=int,
      default=100,
      help="Rolling-mean window in optimizer steps, not saved CSV rows.",
  )
  parser.add_argument("--output-prefix", default="scan_D_d_diagnostics")
  args = parser.parse_args()

  scan_dir = args.scan_dir.resolve()
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
  runs, duplicate_count = _deduplicate_runs(runs)

  dipole_values = sorted({run.dipole_strength for run in runs})
  d_values = sorted({run.d for run in runs})
  component_bars_output = (
      scan_dir / f"{args.output_prefix}_energy_component_bars.png")
  summary_output = scan_dir / f"{args.output_prefix}_summary.png"
  _plot_energy_component_bars(
      runs, dipole_values, d_values, component_bars_output)
  _plot_scan_metric_summary(runs, dipole_values, d_values, summary_output)

  print(f"Loaded {len(runs)} runs from {scan_dir}")
  if duplicate_count:
    print(
        f"Collapsed {duplicate_count} duplicate (D, d) runs by best checkpoint score")
  if skipped:
    print(
        f"Skipped {len(skipped)} runs without usable training stats/components")
  print(f"Saved energy component bars to {component_bars_output}")
  print(f"Saved summary heatmaps to {summary_output}")


if __name__ == "__main__":
  main()
