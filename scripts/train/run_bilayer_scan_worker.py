#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Run one shard of an rs,d scan.

This is designed for MIT SuperCloud LLsub Triples. Each LLsub process receives
LLSUB_RANK and LLSUB_SIZE; this worker runs every LLSUB_SIZE-th row of the scan
manifest starting at LLSUB_RANK.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "scan_manifests" / "rs_d_manifest.csv"
DEFAULT_RUN_SCRIPT = REPO_ROOT / "periodicwave" / "configs" / "bilayer_bosons.py"


def _default_int_from_env(name: str, default: int) -> int:
  value = os.environ.get(name)
  if value is None:
    return default
  return int(value)


def _as_config_float_string(value: str) -> str:
  return str(float(value))


def _as_config_int(value: str) -> int:
  return int(float(value))


def _read_manifest(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise ValueError(f"{path} does not contain any scan points")
  missing = [name for name in ("rs", "d") if name not in rows[0]]
  if missing:
    raise ValueError(f"{path} is missing required columns: {missing}")
  return rows


def _result_glob(rs: str, d: str, seed: int) -> str:
  return str(
      REPO_ROOT
      / "results"
      / "bilayer-bosons"
      / "BosonNet"
      / f"*_rs{rs}_d{d}_D*_seed{seed}_*")


def _max_recorded_step(result_dir: Path) -> int | None:
  stats_files = sorted(result_dir.glob("train_stats*.csv"))
  max_step = None
  for stats_file in stats_files:
    with stats_file.open(newline="", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        if not row.get("step"):
          continue
        step = int(float(row["step"]))
        max_step = step if max_step is None else max(max_step, step)
  return max_step


def _already_complete(rs: str, d: str, seed: int, min_step: int) -> bool:
  for result_name in glob.glob(_result_glob(rs, d, seed)):
    result_dir = Path(result_name)
    if not (result_dir / "config.json").exists():
      continue
    max_step = _max_recorded_step(result_dir)
    if max_step is not None and max_step >= min_step:
      return True
  return False


def _run_point(run_script: Path, rs: str, d: str, seed: int) -> None:
  env = os.environ.copy()
  env["SCAN_RS"] = rs
  env["SCAN_D"] = d
  env["SCAN_SEED"] = str(seed)
  env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
  env.setdefault("XDG_CACHE_HOME", "/tmp")
  old_pythonpath = env.get("PYTHONPATH")
  env["PYTHONPATH"] = (
      str(REPO_ROOT) if not old_pythonpath
      else f"{REPO_ROOT}{os.pathsep}{old_pythonpath}")

  cmd = [sys.executable, str(run_script)]
  print(f"Running rs={rs}, d={d}, seed={seed}: {' '.join(cmd)}", flush=True)
  subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
  parser.add_argument("--rank", type=int,
                      default=_default_int_from_env("LLSUB_RANK", 0))
  parser.add_argument("--size", type=int,
                      default=_default_int_from_env("LLSUB_SIZE", 1))
  parser.add_argument("--min-completed-step", type=int,
                      default=_default_int_from_env(
                          "SCAN_MIN_COMPLETED_STEP", 0))
  parser.add_argument("--seed", type=int,
                      default=_default_int_from_env("SCAN_SEED", 42),
                      help="Deterministic seed for this scan batch.")
  parser.add_argument("--force", action="store_true",
                      help="Run points even if matching results already exist.")
  parser.add_argument("--dry-run", action="store_true",
                      help="Print assigned scan points without running them.")
  args = parser.parse_args()

  if args.size <= 0:
    raise ValueError("--size must be positive")
  if not 0 <= args.rank < args.size:
    raise ValueError("--rank must satisfy 0 <= rank < size")

  rows = _read_manifest(args.manifest)
  assigned_rows = rows[args.rank::args.size]
  print(
      f"Worker rank {args.rank}/{args.size} received "
      f"{len(assigned_rows)} of {len(rows)} scan points.",
      flush=True)

  for row in assigned_rows:
    rs = _as_config_float_string(row["rs"])
    d = _as_config_float_string(row["d"])
    seed = _as_config_int(row["seed"]) if row.get("seed") else args.seed
    if args.dry_run:
      print(f"Would run rs={rs}, d={d}, seed={seed}", flush=True)
      continue
    if not args.force and _already_complete(
        rs, d, seed, args.min_completed_step):
      print(f"Skipping existing rs={rs}, d={d}, seed={seed}", flush=True)
      continue
    _run_point(args.run_script, rs, d, seed)


if __name__ == "__main__":
  main()
