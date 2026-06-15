#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot energy convergence for bilayer boson scan runs."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_PATTERN = "N*_layers*_*_rs*_d*_D*_sq"
RUN_RE = re.compile(
    r"N(?P<num_bosons>\d+)_layers(?P<layer_a>\d+)_(?P<layer_b>\d+)"
    r"_rs(?P<rs>[0-9.]+)_d(?P<d>[0-9.]+)_D(?P<dipole>[0-9.]+)"
    r"(?:_seed(?P<seed>\d+))?_"
    r"(?P<cell>sq|tri)(?:_[^/]+)?$"
)
COLORS = {
    "total": "#2563eb",
    "ewmean": "#111827",
    "kinetic": "#7c3aed",
    "potential": "#dc2626",
    "intra": "#059669",
    "inter": "#d97706",
    "acceptance": "#0f766e",
    "width": "#be123c",
    "esjd": "#0284c7",
    "esjd_moved": "#4f46e5",
}


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


@dataclass(frozen=True)
class BestSelection:
  step: int
  score: float
  metric: str
  row: pd.Series
  source: str


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
      seed=int(parsed["seed"]) if parsed["seed"] is not None else None,
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


def _best_checkpoint_metric(run_dir: Path) -> str:
  config_path = run_dir / "config.json"
  if not config_path.exists():
    return "ewmean"
  with config_path.open(encoding="utf-8") as f:
    config = json.load(f)
  return config.get("log", {}).get("best_checkpoint_metric", "ewmean")


def _best_checkpoint_std_weight(run_dir: Path) -> float:
  config_path = run_dir / "config.json"
  if not config_path.exists():
    return 1.0
  with config_path.open(encoding="utf-8") as f:
    config = json.load(f)
  return float(config.get("log", {}).get("best_checkpoint_std_weight", 1.0))


def _row_at_step(data: pd.DataFrame, step: int) -> pd.Series:
  exact = data[data["step"] == step]
  if not exact.empty:
    return exact.iloc[0]
  nearest = (data["step"] - step).abs().idxmin()
  return data.loc[nearest]


def _best_selection(params: RunParams, plot_data: pd.DataFrame) -> BestSelection:
  metric = _best_checkpoint_metric(params.path)
  manifest_path = params.path / "qmcjax_best_checkpoints.csv"
  if manifest_path.exists() and manifest_path.stat().st_size > 0:
    manifest = pd.read_csv(manifest_path)
    if not manifest.empty and {"step", "score"}.issubset(manifest.columns):
      best = manifest.sort_values(["score", "step"]).iloc[0]
      step = int(best["step"])
      return BestSelection(
          step=step,
          score=float(best["score"]),
          metric=metric,
          row=_row_at_step(plot_data, step),
          source="checkpoint")

  if metric == "ewmean" and "ewmean" in plot_data:
    idx = plot_data["ewmean"].idxmin()
    score = float(plot_data.loc[idx, "ewmean"])
  elif metric == "variance" and "locstd" in plot_data:
    idx = (plot_data["locstd"] ** 2).idxmin()
    score = float(plot_data.loc[idx, "locstd"] ** 2)
  elif metric == "energy_std" and "locstd" in plot_data:
    std_weight = _best_checkpoint_std_weight(params.path)
    scores = plot_data["energy"] + std_weight * plot_data["locstd"]
    idx = scores.idxmin()
    score = float(scores.loc[idx])
  elif metric == "ewmean_std" and {"ewmean", "locstd"}.issubset(plot_data.columns):
    std_weight = _best_checkpoint_std_weight(params.path)
    scores = plot_data["ewmean"] + std_weight * plot_data["locstd"]
    idx = scores.idxmin()
    score = float(scores.loc[idx])
  else:
    metric = "energy"
    idx = plot_data["energy"].idxmin()
    score = float(plot_data.loc[idx, "energy"])
  row = plot_data.loc[idx]
  return BestSelection(
      step=int(row["step"]),
      score=score,
      metric=metric,
      row=row,
      source="train_stats")


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


def _add_rolling_average(ax, x, y, label, rolling_window, color=None):
  ax.plot(
      x,
      y,
      linewidth=0.55,
      alpha=0.28,
      color=color,
      label=f"{label}, logged")
  ax.scatter(
      x,
      y,
      s=7,
      alpha=0.38,
      color=color,
      edgecolors="none",
      rasterized=True)
  rolling_rows = _rolling_rows_from_step_window(x, rolling_window)
  if len(y) >= rolling_rows:
    rolling = y.rolling(
        rolling_rows,
        min_periods=max(1, rolling_rows // 5)).mean()
    ax.plot(
        x,
        rolling,
        linewidth=2.1,
        color=color,
        label=f"{label}, rolling {rolling_window} steps")


def _set_robust_ylim(
    ax,
    series_list,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    include_zero: bool = False,
    positive_only: bool = False,
) -> None:
  values = []
  for series in series_list:
    clean = pd.to_numeric(pd.Series(series), errors="coerce")
    clean = clean.replace([float("inf"), -float("inf")], pd.NA).dropna()
    if positive_only:
      clean = clean[clean > 0.0]
    if not clean.empty:
      values.append(clean)
  if not values:
    return

  data = pd.concat(values, ignore_index=True)
  q1 = float(data.quantile(0.25))
  q3 = float(data.quantile(0.75))
  iqr = q3 - q1
  quantile_lo = float(data.quantile(lower_quantile))
  quantile_hi = float(data.quantile(upper_quantile))
  if iqr > 0.0:
    lo = max(quantile_lo, q1 - 3.0 * iqr)
    hi = min(quantile_hi, q3 + 3.0 * iqr)
  else:
    lo = quantile_lo
    hi = quantile_hi
  if include_zero:
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
  if not hi > lo:
    pad = max(abs(hi), 1.0) * 0.05
    lo -= pad
    hi += pad
  else:
    pad = 0.08 * (hi - lo)
    lo -= pad
    hi += pad
  if positive_only:
    lo = max(lo, float(data.min()) * 0.5, 1.0e-12)
  ax.set_ylim(lo, hi)


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


def _format_run_title(params: RunParams) -> str:
  title = (
      f"rs={params.rs:g}, d={params.d:g}, "
      f"D={params.dipole_strength:g}")
  if params.seed is not None:
    title += f", seed={params.seed}"
  return title


def _score_per_particle(selection: BestSelection, params: RunParams) -> float:
  del params
  return selection.score


def _selection_summary(
    params: RunParams,
    selection: BestSelection,
    include_uncertainty: bool = True,
) -> str:
  lines = [
      f"best step = {selection.step} ({selection.metric})",
      f"best score/N = {_format_float(_score_per_particle(selection, params))}",
      f"E/N at best = {_format_float(selection.row['energy'])}",
  ]
  if not include_uncertainty:
    return "\n".join(lines)
  if "locstd" in selection.row:
    std = selection.row["locstd"]
    lines.append(f"std(E_L)/N at best = {_format_float(std)}")
  elif "ewvar" in selection.row:
    ewstd = selection.row["ewvar"] ** 0.5
    lines.append(f"EW std(E_L)/N at best = {_format_float(ewstd)}")
  return "\n".join(lines)


def _plot_energy(params: RunParams, plot_data: pd.DataFrame) -> None:
  selection = _best_selection(params, plot_data)
  fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
  ax.plot(
      plot_data["step"],
      plot_data["energy"],
      marker="o",
      linestyle="-",
      linewidth=0.4,
      markersize=1,
      alpha=0.35,
      label="energy per boson")
  ax.set_xlabel("step")
  ax.set_ylabel("energy per boson")
  if (plot_data["energy"] > 0).all():
    ax.set_yscale("log")
    _set_robust_ylim(ax, [plot_data["energy"]], positive_only=True)
  else:
    _set_robust_ylim(ax, [plot_data["energy"]])
  ax.set_title(_format_run_title(params))
  _add_text_box(ax, _selection_summary(params, selection))
  ax.legend()
  fig.tight_layout()
  output_path = params.path / "fig_training_energy_trace.png"
  fig.savefig(output_path, dpi=200)
  print(f"Saved {output_path}")
  plt.close(fig)


def _plot_energy_components(
    params: RunParams,
    plot_data: pd.DataFrame,
    rolling_window: int,
) -> None:
  selection = _best_selection(params, plot_data)
  fig, axs = plt.subplots(4, 1, figsize=(10.5, 12.0), sharex=True)
  axs = axs.ravel()
  steps = plot_data["step"]
  component_columns = [
      ("kinetic_energy", "kinetic / N", COLORS["kinetic"]),
      ("potential_intra", "intralayer / N", COLORS["intra"]),
      ("potential_inter", "interlayer / N", COLORS["inter"]),
  ]
  available_component_columns = [
      item for item in component_columns if item[0] in plot_data]

  _add_rolling_average(
      axs[0],
      steps,
      plot_data["energy"],
      "total energy / N",
      rolling_window,
      color=COLORS["total"])
  axs[0].set_ylabel("total / N")
  _set_robust_ylim(axs[0], [plot_data["energy"]])
  _add_text_box(axs[0], _selection_summary(
      params, selection, include_uncertainty=False))
  axs[0].legend(fontsize=8)

  if available_component_columns:
    axs[1].axhline(0.0, color="0.55", linewidth=0.8, alpha=0.8)
    for column, label, color in available_component_columns:
      _add_rolling_average(
          axs[1],
          steps,
          plot_data[column],
          label,
          rolling_window,
          color=color)
    axs[1].set_ylabel("components / N")
    _set_robust_ylim(
        axs[1],
        [plot_data[column] for column, _, _ in available_component_columns],
        include_zero=True)
    axs[1].legend(fontsize=8, ncol=2)
  else:
    axs[1].text(0.5, 0.5, "energy components unavailable",
                ha="center", va="center")

  if available_component_columns:
    denominator = sum(
        plot_data[column].abs()
        for column, _, _ in available_component_columns)
    denominator = denominator.where(denominator > 1.0e-12)
    axs[2].axhline(1.0, color="0.55", linewidth=0.8, alpha=0.5)
    for column, label, color in available_component_columns:
      _add_rolling_average(
          axs[2],
          steps,
          plot_data[column].abs() / denominator,
          "|" + label.replace(" / N", "| fraction"),
          rolling_window,
          color=color)
    axs[2].set_ylim(-0.03, 1.03)
    axs[2].set_ylabel("|component| fraction")
    axs[2].legend(fontsize=8, ncol=2)
  else:
    axs[2].text(0.5, 0.5, "energy ratios unavailable",
                ha="center", va="center")

  if "locstd" in plot_data:
    std_per_particle = plot_data["locstd"]
    _add_rolling_average(
        axs[3],
        steps,
        std_per_particle,
        "std(E_L) / N",
        rolling_window,
        color="#64748b")
    axs[3].set_ylabel("locstd / N")
    if (std_per_particle > 0).all():
      axs[3].set_yscale("log")
      _set_robust_ylim(axs[3], [std_per_particle], positive_only=True)
    else:
      _set_robust_ylim(axs[3], [std_per_particle])
    std_at_best = selection.row["locstd"]
    _add_text_box(
        axs[3],
        f"std(E_L)/N at best = {_format_float(std_at_best)}")
    axs[3].legend(fontsize=8)
  elif "ewvar" in plot_data:
    ewstd_per_particle = plot_data["ewvar"] ** 0.5
    _add_rolling_average(
        axs[3],
        steps,
        ewstd_per_particle,
        "EW std(E_L) / N",
        rolling_window,
        color="#64748b")
    axs[3].set_ylabel("EW std / N")
    if (ewstd_per_particle > 0).all():
      axs[3].set_yscale("log")
      _set_robust_ylim(axs[3], [ewstd_per_particle], positive_only=True)
    else:
      _set_robust_ylim(axs[3], [ewstd_per_particle])
    axs[3].legend(fontsize=8)
  else:
    axs[3].text(0.5, 0.5, "locstd unavailable",
                ha="center", va="center")

  for ax in axs:
    ax.grid(alpha=0.25, linewidth=0.5)
  axs[-1].set_xlabel("step")

  fig.suptitle(
      f"Energy diagnostics: {_format_run_title(params)}",
      y=0.995)
  fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
  output_path = params.path / "fig_training_energy_components.png"
  fig.savefig(output_path, dpi=200)
  print(f"Saved {output_path}")
  plt.close(fig)


def _plot_mcmc_diagnostics(
    params: RunParams,
    plot_data: pd.DataFrame,
    rolling_window: int,
) -> None:
  fig, axs = plt.subplots(4, 1, figsize=(9.5, 11.5), sharex=True)
  axs = axs.ravel()
  steps = plot_data["step"]

  if "pmove" in plot_data:
    _add_rolling_average(
        axs[0],
        steps,
        plot_data["pmove"],
        "pmove",
        rolling_window,
        color=COLORS["acceptance"])
    axs[0].set_ylabel("acceptance")
    axs[0].set_ylim(0.0, 1.0)
    axs[0].legend(fontsize=8)
  else:
    axs[0].text(0.5, 0.5, "pmove unavailable", ha="center", va="center")

  if "mcmc_width" in plot_data:
    _add_rolling_average(
        axs[1],
        steps,
        plot_data["mcmc_width"],
        "MCMC move width",
        rolling_window,
        color=COLORS["width"])
    axs[1].set_ylabel("move width")
    if (plot_data["mcmc_width"] > 0).all():
      axs[1].set_yscale("log")
    _add_text_box(
        axs[1],
        f"final width = {_format_float(plot_data['mcmc_width'].iloc[-1])}")
    axs[1].legend(fontsize=8)
  else:
    axs[1].text(0.5, 0.5, "mcmc_width unavailable", ha="center", va="center")

  if "mcmc_esjd_per_particle" in plot_data:
    _add_rolling_average(
        axs[2],
        steps,
        plot_data["mcmc_esjd_per_particle"],
        "ESJD / particle",
        rolling_window,
        color=COLORS["esjd"])
    axs[2].set_ylabel("ESJD / particle")
    if (plot_data["mcmc_esjd_per_particle"] > 0).all():
      axs[2].set_yscale("log")
    axs[2].legend(fontsize=8)
  else:
    axs[2].text(
        0.5, 0.5, "mcmc_esjd_per_particle unavailable",
        ha="center", va="center")

  if "mcmc_esjd_per_moved_particle" in plot_data:
    _add_rolling_average(
        axs[3],
        steps,
        plot_data["mcmc_esjd_per_moved_particle"],
        "ESJD / moved particle",
        rolling_window,
        color=COLORS["esjd_moved"])
    axs[3].set_ylabel("ESJD / moved")
    if (plot_data["mcmc_esjd_per_moved_particle"] > 0).all():
      axs[3].set_yscale("log")
    axs[3].legend(fontsize=8)
  else:
    axs[3].text(
        0.5, 0.5, "mcmc_esjd_per_moved_particle unavailable",
        ha="center", va="center")

  for ax in axs:
    ax.grid(alpha=0.25, linewidth=0.5)
  axs[-1].set_xlabel("step")

  fig.suptitle(
      f"MCMC diagnostics: {_format_run_title(params)}",
      y=0.995)
  fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
  output_path = params.path / "fig_training_mcmc_diagnostics.png"
  fig.savefig(output_path, dpi=200)
  print(f"Saved {output_path}")
  plt.close(fig)


def _evaluate_run(
    run_dir: Path,
    burn_in_cut: int,
    rolling_window: int,
    skip_existing: bool,
    write_extra_figures: bool = True,
) -> None:
  required_outputs = [
      run_dir / "fig_training_energy_components.png",
      run_dir / "fig_training_mcmc_diagnostics.png",
  ]
  if write_extra_figures:
    required_outputs.append(run_dir / "fig_training_energy_trace.png")
  if skip_existing and all(path.exists() for path in required_outputs):
    print(f"Skipping existing energy plots: {run_dir}")
    return

  params = _parse_run_dir(run_dir)
  train_data = _load_train_stats(run_dir)
  if "step" not in train_data or "energy" not in train_data:
    raise ValueError(f"{run_dir} train stats must contain step and energy columns")
  plot_data = train_data[train_data["step"] >= burn_in_cut].reset_index(drop=True)
  if plot_data.empty:
    print(
        f"No rows at or after step {burn_in_cut} in {run_dir.name}; plotting "
        "all rows instead.")
    plot_data = train_data.reset_index(drop=True)
  if write_extra_figures:
    _plot_energy(params, plot_data)
  _plot_energy_components(params, plot_data, rolling_window)
  _plot_mcmc_diagnostics(params, plot_data, rolling_window)


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
  parser.add_argument(
      "--burn-in-cut",
      type=int,
      default=0,
      help="Drop rows with training step smaller than this value.",
  )
  parser.add_argument(
      "--rolling-window",
      type=int,
      default=100,
      help="Rolling-mean window in optimizer steps, not saved CSV rows.",
  )
  parser.add_argument(
      "--skip-existing",
      action="store_true",
      help="Skip runs where the requested training output plots already exist.",
  )
  parser.add_argument(
      "--write-extra-figures",
      action="store_true",
      help="Also write the standalone energy trace figure.",
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
          args.write_extra_figures,
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
