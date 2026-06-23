# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Shared helpers for bilayer scan visualization scripts."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path

from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

from periodicwave.pbc import lattices


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERN = "N*_layers*_*_rs*_d*_D*_sq"
DEFAULT_D_D_SCAN_DIR = REPO_ROOT / "results" / "D_d_scan_N24_rs1.0_sq"
DEFAULT_RS_D_SCAN_DIR = (
    REPO_ROOT / "results" / "bilayer-bosons" / "BosonNet" / "scan_260603"
)
RUN_RE = re.compile(
    r"N(?P<num_bosons>\d+)_layers(?P<layer_a>\d+)_(?P<layer_b>\d+)"
    r"_rs(?P<rs>[0-9.]+)_d(?P<d>[0-9.]+)_D(?P<dipole>[0-9.]+)"
    r"(?:_seed(?P<seed>\d+))?_"
    r"(?P<cell>sq|tri)(?:_[^/]+)?$"
)


def normalize_x_param(x_param: str) -> str:
  if x_param == "D":
    return x_param
  if x_param.lower() == "rs":
    return "rs"
  raise ValueError(f"Unknown x-param {x_param!r}; expected 'rs' or 'D'.")


def x_attr(x_param: str) -> str:
  x_param = normalize_x_param(x_param)
  return "dipole_strength" if x_param == "D" else "rs"


def x_label(x_param: str) -> str:
  return normalize_x_param(x_param)


def x_value(run_or_params, x_param: str) -> float:
  return float(getattr(run_or_params, x_attr(x_param)))


def scan_name(x_param: str) -> str:
  return f"{x_label(x_param)},d"


def default_scan_dir(x_param: str) -> Path:
  return DEFAULT_D_D_SCAN_DIR if normalize_x_param(x_param) == "D" else (
      DEFAULT_RS_D_SCAN_DIR)


def default_output_prefix(kind: str, x_param: str) -> str:
  return f"scan_{x_label(x_param)}_d_{kind}"


def parse_run_dir(run_dir: Path) -> dict[str, str]:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  return match.groupdict()


def read_config(run_dir: Path) -> dict:
  config_path = run_dir / "config.json"
  if not config_path.exists():
    return {}
  with config_path.open(encoding="utf-8") as f:
    return json.load(f)


def num_bosons(run_dir: Path, parsed: dict[str, str]) -> int:
  config = read_config(run_dir)
  config_bosons = config.get("system", {}).get("bosons", [None])[0]
  if config_bosons:
    return int(config_bosons)
  return int(parsed["num_bosons"])


def load_train_stats(run_dir: Path) -> pd.DataFrame | None:
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


def best_checkpoint_record(run_dir: Path) -> tuple[int | None, float]:
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


def nearest_step_row(stats: pd.DataFrame, step: int | None) -> pd.Series:
  if step is None:
    return stats.iloc[-1]
  idx = (stats["step"] - step).abs().idxmin()
  return stats.loc[idx]


def rolling_rows_from_step_window(
    steps: pd.Series,
    rolling_window: int,
) -> int:
  """Converts a requested optimizer-step window to saved CSV rows."""
  if rolling_window <= 0 or len(steps) < 2:
    return max(1, rolling_window)
  step_diffs = steps.sort_values().diff().dropna()
  step_diffs = step_diffs[step_diffs > 0]
  if step_diffs.empty:
    return max(1, rolling_window)
  step_interval = float(step_diffs.median())
  return max(1, int(round(float(rolling_window) / step_interval)))


def latest_checkpoint_files(run_dir: Path, nfiles: int) -> list[Path]:
  numbered = []
  for path in run_dir.glob("qmcjax_ckpt_*.npz"):
    try:
      numbered.append((int(path.stem.split("_")[-1]), path))
    except ValueError:
      pass
  return [path for _, path in sorted(numbered, reverse=True)[:nfiles]]


def best_checkpoint_files(run_dir: Path, nfiles: int) -> list[Path]:
  manifest_path = run_dir / "qmcjax_best_checkpoints.csv"
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
    checkpoint_path = run_dir / checkpoint_name
    if checkpoint_path.exists():
      candidates.append((score, checkpoint_path))
  candidates.sort(key=lambda item: item[0])
  return [path for _, path in candidates[:nfiles]]


def checkpoint_files(run_dir: Path, nfiles: int) -> list[Path]:
  best_files = best_checkpoint_files(run_dir, nfiles)
  if best_files:
    return best_files
  return latest_checkpoint_files(run_dir, nfiles)


def run_selection_score(run_dir: Path) -> tuple[float, int, str]:
  manifest_path = run_dir / "qmcjax_best_checkpoints.csv"
  if manifest_path.exists() and manifest_path.stat().st_size > 0:
    rows = np.genfromtxt(
        manifest_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8")
    rows = np.atleast_1d(rows)
    candidates = []
    for row in rows:
      try:
        score = float(row["score"])
        step = int(row["step"])
        checkpoint_path = run_dir / str(row["checkpoint"])
      except (ValueError, KeyError):
        continue
      if checkpoint_path.exists():
        candidates.append((score, step, run_dir.name))
    if candidates:
      return min(candidates)

  latest = latest_checkpoint_files(run_dir, 1)
  if latest:
    try:
      return (float("inf"), -int(latest[0].stem.split("_")[-1]), run_dir.name)
    except ValueError:
      pass
  return (float("inf"), 0, run_dir.name)


def install_jax_array_unpickle_fallback() -> None:
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


def lattice(
    num_bosons_: int,
    density_rs: float,
    supercell_shape: str,
) -> np.ndarray:
  if supercell_shape == "tri":
    supercell_a = density_rs * np.sqrt(2 * np.pi / np.sqrt(3) * num_bosons_)
    lat_vec, _ = lattices._triangular_lattice_vecs_periodic_potential(
        supercell_a, 1)
    return lat_vec
  if supercell_shape == "sq":
    supercell_a = density_rs * np.sqrt(np.pi * num_bosons_)
    return lattices._square_lattice_vecs(supercell_a)
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")


def finite_norm(values: np.ndarray) -> Normalize:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return Normalize(vmin=0.0, vmax=1.0)
  vmin = float(finite.min())
  vmax = float(finite.max())
  if math.isclose(vmin, vmax):
    return Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
  return Normalize(vmin=vmin, vmax=vmax)


def finite_percentile_norm(
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


def finite_centered_norm(values: np.ndarray, center: float = 0.0) -> Normalize:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return TwoSlopeNorm(vmin=-1.0, vcenter=center, vmax=1.0)
  vmin, vmax = np.percentile(finite, [5.0, 95.0])
  vmin = min(float(vmin), center - 1e-6)
  vmax = max(float(vmax), center + 1e-6)
  return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def text_color_for_value(cmap, norm: Normalize, value: float) -> str:
  normalized = norm(value)
  if np.ma.is_masked(normalized):
    return "black"
  red, green, blue, _ = cmap(float(normalized))
  luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
  return "black" if luminance > 0.58 else "white"


def draw_missing_panel(ax) -> None:
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


def reversed_d_values(d_values: list[float]) -> list[float]:
  return list(reversed(d_values))
