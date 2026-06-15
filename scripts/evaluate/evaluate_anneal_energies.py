#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot a continuous energy trace across annealed bilayer runs."""

from __future__ import annotations

import argparse
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
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_DIR = REPO_ROOT / "results" / "anneal_rs"
DEFAULT_PATTERN = "N24_layers12_12_rs*_d*_D20.0_seed*_sq"
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
  rs: float
  d: float
  dipole_strength: float
  seed: int | None


def _parse_run_dir(run_dir: Path) -> RunParams:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  parsed = match.groupdict()
  return RunParams(
      path=run_dir,
      num_bosons=int(parsed["num_bosons"]),
      rs=float(parsed["rs"]),
      d=float(parsed["d"]),
      dipole_strength=float(parsed["dipole"]),
      seed=int(parsed["seed"]) if parsed["seed"] else None,
  )


def _load_train_stats(folder_path: Path) -> pd.DataFrame:
  stats_files = [folder_path / "train_stats.csv"] + sorted(
      folder_path.glob("train_stats_*.csv"))
  frames = []
  for stats_file in stats_files:
    if stats_file.exists() and stats_file.stat().st_size > 0:
      data = pd.read_csv(stats_file)
      if not data.empty:
        frames.append(data)
  if not frames:
    raise ValueError(f"No non-empty train_stats*.csv found in {folder_path}")
  data = pd.concat(frames, ignore_index=True)
  if "step" in data:
    data = data.sort_values("step").reset_index(drop=True)
  return data


def _select_run_dirs(scan_dir: Path, pattern: str) -> list[Path]:
  run_dirs = []
  for run_dir in sorted(scan_dir.glob(pattern)):
    if not run_dir.is_dir():
      continue
    if not (run_dir / "config.json").exists():
      continue
    _parse_run_dir(run_dir)
    run_dirs.append(run_dir)
  if not run_dirs:
    raise ValueError(f"No run directories matched {scan_dir / pattern}")
  return run_dirs


def _sort_runs(
    runs: list[tuple[RunParams, pd.DataFrame]],
    order: str,
) -> list[tuple[RunParams, pd.DataFrame]]:
  reverse = order == "rs-desc"
  if order in ("rs-asc", "rs-desc"):
    return sorted(runs, key=lambda item: item[0].rs, reverse=reverse)
  if order == "path":
    return sorted(runs, key=lambda item: item[0].path.name)
  raise ValueError(f"Unknown order: {order}")


def _combined_trace(
    runs: list[tuple[RunParams, pd.DataFrame]],
) -> tuple[pd.DataFrame, list[tuple[int, RunParams]]]:
  frames = []
  boundaries = []
  offset = 0
  for stage, (params, data) in enumerate(runs):
    if "step" not in data:
      raise ValueError(f"{params.path} is missing a step column")
    stage_data = data.copy()
    stage_data["stage"] = stage
    stage_data["local_step"] = stage_data["step"]
    stage_data["global_step"] = stage_data["local_step"] + offset
    stage_data["rs"] = params.rs
    stage_data["d"] = params.d
    stage_data["seed"] = params.seed
    stage_data["num_bosons"] = params.num_bosons
    stage_data["energy_per_particle"] = (
        stage_data["energy"] / params.num_bosons)
    if "ewmean" in stage_data:
      stage_data["ewmean_per_particle"] = (
          stage_data["ewmean"] / params.num_bosons)
    if "locstd" in stage_data:
      stage_data["variance_per_particle"] = (
          stage_data["locstd"] / params.num_bosons) ** 2
    elif "ewvar" in stage_data:
      stage_data["ewvar_per_particle"] = (
          stage_data["ewvar"] / (params.num_bosons ** 2))
    frames.append(stage_data)
    boundaries.append((offset, params))
    offset += int(stage_data["local_step"].max()) + 1
  return pd.concat(frames, ignore_index=True), boundaries


def _format_float(value: float) -> str:
  return f"{value:.6g}"


def _add_text_box(ax, text: str) -> None:
  ax.text(
      0.03,
      0.97,
      text,
      transform=ax.transAxes,
      ha="left",
      va="top",
      fontsize=8,
      bbox={
          "boxstyle": "round,pad=0.25",
          "facecolor": "white",
          "edgecolor": "0.75",
          "alpha": 0.85,
      })


def _add_rolling_average(ax, x, y, label, rolling_window, color=None):
  ax.plot(
      x,
      y,
      marker="o",
      linestyle="-",
      linewidth=0.4,
      markersize=1,
      alpha=0.25,
      color=color,
      label=label)
  if len(y) >= rolling_window:
    rolling = y.rolling(
        rolling_window,
        min_periods=max(1, rolling_window // 5)).mean()
    ax.plot(
        x,
        rolling,
        linewidth=1.8,
        color=color,
        label=f"{label}, rolling mean")


def _stage_label_indices(count: int) -> set[int]:
  if count <= 8:
    return set(range(count))
  return {0, count - 1} | set(range(2, count - 1, 3))


def _add_stage_markers(
    axs,
    trace: pd.DataFrame,
    boundaries: list[tuple[int, RunParams]],
    *,
    label: bool = True,
) -> None:
  max_step = max(float(trace["global_step"].max()), 1.0)
  label_indices = _stage_label_indices(len(boundaries))
  for ax in axs:
    ymin, ymax = ax.get_ylim()
    label_y = ymax - 0.04 * (ymax - ymin)
    for index, (boundary, params) in enumerate(boundaries):
      ax.axvline(
          boundary, color="0.35", linestyle="--", linewidth=0.8, alpha=0.6)
      if not label or index not in label_indices:
        continue
      ax.text(
          boundary + 0.01 * max_step,
          label_y,
          f"rs={params.rs:g}",
          rotation=90,
          va="top",
          ha="left",
          fontsize=8,
          color="0.25")


def _plot_trace(
    trace: pd.DataFrame,
    boundaries: list[tuple[int, RunParams]],
    output_path: Path,
) -> None:
  fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
  ax.plot(
      trace["global_step"],
      trace["energy_per_particle"],
      marker="o",
      linestyle="-",
      linewidth=0.5,
      markersize=1,
      alpha=0.35,
      label="energy / N")

  first = boundaries[0][1]
  last = boundaries[-1][1]
  ax.set_xlabel("annealing step")
  ax.set_ylabel("energy per boson")
  ax.set_title(
      f"Annealed energy trace, d={first.d:g}, "
      f"rs {first.rs:g} -> {last.rs:g}")
  ax.legend()
  ax.grid(alpha=0.2, linewidth=0.5)
  _add_stage_markers([ax], trace, boundaries)

  final_energy = trace["energy_per_particle"].iloc[-1]
  ax.text(
      0.99,
      0.03,
      f"final E/N = {_format_float(final_energy)}",
      transform=ax.transAxes,
      ha="right",
      va="bottom",
      fontsize=9,
      bbox={
          "boxstyle": "round,pad=0.25",
          "facecolor": "white",
          "edgecolor": "0.75",
          "alpha": 0.85,
      })

  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  print(f"Saved {output_path}")
  plt.close(fig)


def _plot_diagnostics(
    trace: pd.DataFrame,
    boundaries: list[tuple[int, RunParams]],
    output_path: Path,
    rolling_window: int,
) -> None:
  fig, axs = plt.subplots(4, 1, figsize=(9.5, 13), sharex=True)
  axs = axs.ravel()
  steps = trace["global_step"]

  _add_rolling_average(
      axs[0],
      steps,
      trace["energy_per_particle"],
      "energy / N",
      rolling_window,
      color="#2a6fbb")
  axs[0].set_ylabel("energy / N")
  if (trace["energy_per_particle"] > 0).all():
    axs[0].set_yscale("log")
  _add_text_box(
      axs[0],
      f"final E/N = {_format_float(trace['energy_per_particle'].iloc[-1])}")
  axs[0].legend(fontsize=8)

  if "variance_per_particle" in trace:
    _add_rolling_average(
        axs[1],
        steps,
        trace["variance_per_particle"],
        "var(E_L / N)",
        rolling_window,
        color="#2f7d57")
    axs[1].set_ylabel("var(E_L / N)")
    if (trace["variance_per_particle"] > 0).all():
      axs[1].set_yscale("log")
    _add_text_box(
        axs[1],
        f"final var(E/N) = "
        f"{_format_float(trace['variance_per_particle'].iloc[-1])}")
    axs[1].legend(fontsize=8)
  elif "ewvar_per_particle" in trace:
    _add_rolling_average(
        axs[1],
        steps,
        trace["ewvar_per_particle"],
        "EW var(E_L / N)",
        rolling_window,
        color="#2f7d57")
    axs[1].set_ylabel("EW var(E_L / N)")
    if (trace["ewvar_per_particle"] > 0).all():
      axs[1].set_yscale("log")
    _add_text_box(
        axs[1],
        f"final EW var(E/N) = "
        f"{_format_float(trace['ewvar_per_particle'].iloc[-1])}")
    axs[1].legend(fontsize=8)
  else:
    axs[1].text(0.5, 0.5, "variance unavailable", ha="center", va="center")

  if "mcmc_width" in trace:
    _add_rolling_average(
        axs[2],
        steps,
        trace["mcmc_width"],
        "MCMC move width",
        rolling_window,
        color="#c7364f")
    axs[2].set_ylabel("MCMC move width")
    if (trace["mcmc_width"] > 0).all():
      axs[2].set_yscale("log")
    _add_text_box(
        axs[2],
        f"final width = {_format_float(trace['mcmc_width'].iloc[-1])}")
    axs[2].legend(fontsize=8)
  else:
    axs[2].text(0.5, 0.5, "mcmc_width unavailable", ha="center", va="center")

  if "pmove" in trace:
    _add_rolling_average(
        axs[3],
        steps,
        trace["pmove"],
        "pmove",
        rolling_window,
        color="#a36f00")
    axs[3].set_ylabel("MCMC acceptance")
    axs[3].set_ylim(0.0, 1.0)
    axs[3].legend(fontsize=8)
  else:
    axs[3].text(0.5, 0.5, "pmove unavailable", ha="center", va="center")

  _add_stage_markers(axs, trace, boundaries, label=False)
  for ax in axs:
    ax.grid(alpha=0.25, linewidth=0.5)
  axs[-1].set_xlabel("annealing step")

  stage_summary = ", ".join(
      f"{params.rs:g}" for _, params in boundaries)
  fig.text(
      0.5,
      0.015,
      f"stage order rs: {stage_summary}",
      ha="center",
      va="bottom",
      fontsize=9)

  first = boundaries[0][1]
  last = boundaries[-1][1]
  fig.suptitle(
      f"Annealed training diagnostics: d={first.d:g}, "
      f"rs {first.rs:g} -> {last.rs:g}",
      y=0.995)
  fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.985))
  fig.savefig(output_path, dpi=200)
  print(f"Saved {output_path}")
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
  parser.add_argument("--pattern", default=DEFAULT_PATTERN)
  parser.add_argument(
      "--order",
      choices=("rs-desc", "rs-asc", "path"),
      default="rs-desc",
      help="Stage order for the continuous trace.")
  parser.add_argument("--output-prefix", default="anneal_energy_trace")
  parser.add_argument("--rolling-window", type=int, default=100)
  args = parser.parse_args()

  run_dirs = _select_run_dirs(args.scan_dir, args.pattern)
  runs = [(_parse_run_dir(run_dir), _load_train_stats(run_dir))
          for run_dir in run_dirs]
  runs = _sort_runs(runs, args.order)
  trace, boundaries = _combined_trace(runs)

  args.scan_dir.mkdir(parents=True, exist_ok=True)
  csv_path = args.scan_dir / f"{args.output_prefix}.csv"
  png_path = args.scan_dir / f"{args.output_prefix}.png"
  diagnostics_path = args.scan_dir / f"{args.output_prefix}_diagnostics.png"
  trace.to_csv(csv_path, index=False)
  print(f"Saved {csv_path}")
  _plot_trace(trace, boundaries, png_path)
  _plot_diagnostics(trace, boundaries, diagnostics_path, args.rolling_window)


if __name__ == "__main__":
  main()
