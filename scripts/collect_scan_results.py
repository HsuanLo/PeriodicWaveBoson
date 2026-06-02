#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Collect rs,d scan outputs into one CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "bilayer-bosons" / "BosonNet"
DEFAULT_OUTPUT = REPO_ROOT / "scans" / "rs_d_summary.csv"


def _load_train_stats(result_dir: Path) -> pd.DataFrame | None:
  stats_files = [result_dir / "train_stats.csv"] + sorted(
      result_dir.glob("train_stats_*.csv"))
  frames = []
  for stats_file in stats_files:
    if stats_file.exists() and stats_file.stat().st_size > 0:
      frames.append(pd.read_csv(stats_file))
  if not frames:
    return None
  return pd.concat(frames, ignore_index=True)


def _read_config(result_dir: Path) -> dict:
  config_path = result_dir / "config.json"
  if not config_path.exists():
    return {}
  with config_path.open(encoding="utf-8") as f:
    return json.load(f)


def _parse_result_name(result_dir: Path) -> dict[str, str]:
  parts = result_dir.name.split("_")
  values = {}
  for part in parts:
    if part.startswith("rs"):
      values["rs"] = part.removeprefix("rs")
    elif part.startswith("d") and part != "D":
      values["d"] = part.removeprefix("d")
    elif part.startswith("D"):
      values["dipole_strength"] = part.removeprefix("D")
  return values


def _summarize_result(result_dir: Path, rolling_window: int) -> dict:
  stats = _load_train_stats(result_dir)
  config = _read_config(result_dir)
  parsed = _parse_result_name(result_dir)

  row = {
      "path": str(result_dir.relative_to(REPO_ROOT)),
      "status": "missing_stats" if stats is None else "ok",
      "rs": parsed.get("rs", ""),
      "d": parsed.get("d", ""),
      "dipole_strength": parsed.get("dipole_strength", ""),
      "num_rows": 0 if stats is None else len(stats),
      "final_step": "",
      "final_energy_per_N": "",
      "best_rolling_energy_per_N": "",
      "final_pmove": "",
      "final_locstd_per_N": "",
  }

  num_bosons = config.get("system", {}).get("bosons", [None])[0]
  if stats is None or stats.empty or not num_bosons:
    return row

  stats = stats.sort_values("step")
  energy_per_n = stats["energy"] / num_bosons
  rolling = energy_per_n.rolling(
      rolling_window, min_periods=max(1, rolling_window // 5)).mean()
  last = stats.iloc[-1]

  row["final_step"] = int(last["step"])
  row["final_energy_per_N"] = float(energy_per_n.iloc[-1])
  row["best_rolling_energy_per_N"] = float(rolling.min())
  if "pmove" in stats:
    row["final_pmove"] = float(last["pmove"])
  if "locstd" in stats:
    row["final_locstd_per_N"] = float(last["locstd"] / num_bosons)
  return row


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--rolling-window", type=int, default=100)
  args = parser.parse_args()

  result_dirs = sorted(
      path for path in args.results_root.glob("*_rs*_d*_D*") if path.is_dir())
  rows = [
      _summarize_result(result_dir, args.rolling_window)
      for result_dir in result_dirs
  ]

  args.output.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
      "rs",
      "d",
      "dipole_strength",
      "status",
      "num_rows",
      "final_step",
      "final_energy_per_N",
      "best_rolling_energy_per_N",
      "final_pmove",
      "final_locstd_per_N",
      "path",
  ]
  with args.output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
  main()
