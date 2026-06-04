#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot energy convergence for bilayer boson scan runs."""

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
DEFAULT_SCAN_DIR = (
    REPO_ROOT
    / "results"
    / "bilayer-bosons"
    / "BosonNet"
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


def _parse_run_dir(run_dir: Path) -> RunParams:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  parsed = match.groupdict()
  return RunParams(
      path=run_dir,
      num_bosons=int(parsed["num_bosons"]),
      layer_occupations=(int(parsed["layer_a"]), int(parsed["layer_b"])),
      rs=float(parsed["rs"]),
      d=float(parsed["d"]),
      dipole_strength=float(parsed["dipole"]),
      supercell_shape=parsed["cell"],
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
    rolling = y.rolling(rolling_window, min_periods=max(1, rolling_window // 5)).mean()
    ax.plot(
        x,
        rolling,
        linewidth=1.8,
        color=color,
        label=f"{label}, rolling mean")


def _plot_energy(params: RunParams, plot_data: pd.DataFrame) -> None:
  fig, ax = plt.subplots(1, 1, figsize=(7, 5))
  ax.plot(
      plot_data["step"],
      plot_data["energy"] / params.num_bosons,
      marker="o",
      linestyle="-",
      linewidth=0.4,
      markersize=1,
      alpha=0.35,
      label="energy per boson")
  if "ewmean" in plot_data:
    ax.plot(
        plot_data["step"],
        plot_data["ewmean"] / params.num_bosons,
        linewidth=1.6,
        label="weighted mean per boson")
  ax.set_xlabel("step")
  ax.set_ylabel("energy per boson")
  ax.set_title(f"rs={params.rs:g}, d={params.d:g}, D={params.dipole_strength:g}")
  ax.legend()
  fig.tight_layout()
  output_path = params.path / "fig_training_energy_trace.png"
  fig.savefig(output_path, dpi=200)
  print(f"Saved {output_path}")
  plt.close(fig)


def _plot_training_diagnostics(
    params: RunParams,
    plot_data: pd.DataFrame,
    rolling_window: int,
) -> None:
  fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
  axs = axs.ravel()
  steps = plot_data["step"]
  energy_per_particle = plot_data["energy"] / params.num_bosons

  _add_rolling_average(
      axs[0],
      steps,
      energy_per_particle,
      "energy / N",
      rolling_window,
      color="#2a6fbb")
  if "ewmean" in plot_data:
    axs[0].plot(
        steps,
        plot_data["ewmean"] / params.num_bosons,
        linewidth=1.5,
        color="#111111",
        label="EW mean / N")
  axs[0].set_ylabel("energy / N")
  if (energy_per_particle > 0).all():
    axs[0].set_yscale("log")
  axs[0].legend(fontsize=8)

  if "locstd" in plot_data:
    locstd_per_particle = plot_data["locstd"] / params.num_bosons
    _add_rolling_average(
        axs[1],
        steps,
        locstd_per_particle,
        "std(E_L) / N",
        rolling_window,
        color="#c7364f")
    axs[1].set_ylabel("std(E_L) / N")
    if (locstd_per_particle > 0).all():
      axs[1].set_yscale("log")
    axs[1].legend(fontsize=8)
  else:
    axs[1].text(0.5, 0.5, "locstd unavailable", ha="center", va="center")

  if "locstd" in plot_data:
    variance_per_particle = (plot_data["locstd"] / params.num_bosons) ** 2
    _add_rolling_average(
        axs[2],
        steps,
        variance_per_particle,
        "var(E_L / N)",
        rolling_window,
        color="#2f7d57")
    axs[2].set_ylabel("var(E_L / N)")
    if (variance_per_particle > 0).all():
      axs[2].set_yscale("log")
    axs[2].legend(fontsize=8)
  elif "ewvar" in plot_data:
    ewvar_per_particle = plot_data["ewvar"] / (params.num_bosons ** 2)
    _add_rolling_average(
        axs[2],
        steps,
        ewvar_per_particle,
        "EW var(E_L / N)",
        rolling_window,
        color="#2f7d57")
    axs[2].set_ylabel("EW var(E_L / N)")
    if (ewvar_per_particle > 0).all():
      axs[2].set_yscale("log")
    axs[2].legend(fontsize=8)
  else:
    axs[2].text(0.5, 0.5, "variance unavailable", ha="center", va="center")

  if "pmove" in plot_data:
    _add_rolling_average(
        axs[3],
        steps,
        plot_data["pmove"],
        "pmove",
        rolling_window,
        color="#a36f00")
    axs[3].set_ylabel("MCMC acceptance")
    axs[3].set_ylim(0.0, 1.0)
    axs[3].legend(fontsize=8)
  else:
    axs[3].text(0.5, 0.5, "pmove unavailable", ha="center", va="center")

  for ax in axs:
    ax.set_xlabel("step")
    ax.grid(alpha=0.25, linewidth=0.5)

  fig.suptitle(
      f"Training diagnostics: rs={params.rs:g}, d={params.d:g}, "
      f"D={params.dipole_strength:g}",
      y=0.995)
  fig.tight_layout()
  output_path = params.path / "fig_training_diagnostics_overview.png"
  fig.savefig(output_path, dpi=200)
  print(f"Saved {output_path}")
  plt.close(fig)


def _evaluate_run(
    run_dir: Path,
    burn_in_cut: int,
    rolling_window: int,
    skip_existing: bool,
) -> None:
  if skip_existing and (run_dir / "fig_training_energy_trace.png").exists() and (
      run_dir / "fig_training_diagnostics_overview.png").exists():
    print(f"Skipping existing energy plots: {run_dir}")
    return

  params = _parse_run_dir(run_dir)
  train_data = _load_train_stats(run_dir)
  if "step" not in train_data or "energy" not in train_data:
    raise ValueError(f"{run_dir} train stats must contain step and energy columns")
  if len(train_data) <= burn_in_cut:
    print(
        f"Only {len(train_data)} rows found in {run_dir.name}; plotting all rows "
        f"instead of applying cut {burn_in_cut}.")
    burn_in_cut = 0
  plot_data = train_data.iloc[burn_in_cut:].reset_index(drop=True)
  _plot_energy(params, plot_data)
  _plot_training_diagnostics(params, plot_data, rolling_window)


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
  parser.add_argument("--burn-in-cut", type=int, default=0)
  parser.add_argument("--rolling-window", type=int, default=100)
  parser.add_argument(
      "--skip-existing",
      action="store_true",
      help="Skip runs where both output plots already exist.",
  )
  args = parser.parse_args()

  run_dirs = _select_run_dirs(args.run_dir, args.scan_dir, args.pattern)

  failures = []
  for idx, run_dir in enumerate(run_dirs, start=1):
    print(f"Evaluating energy {idx}/{len(run_dirs)}: {run_dir.name}", flush=True)
    try:
      _evaluate_run(
          run_dir,
          args.burn_in_cut,
          args.rolling_window,
          args.skip_existing,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failures.append((run_dir, exc))
      print(f"Failed {run_dir.name}: {exc}", flush=True)

  print(f"Processed {len(run_dirs) - len(failures)}/{len(run_dirs)} energy runs")
  if failures:
    print(f"Failures: {len(failures)}")
    for path, exc in failures[:10]:
      print(f"  {path.name}: {exc}")


if __name__ == "__main__":
  main()
