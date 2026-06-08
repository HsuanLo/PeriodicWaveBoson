#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Run density continuation for the bilayer boson calculation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_SCRIPT = REPO_ROOT / "periodicwave" / "configs" / "bilayer_bosons.py"


def _float_string(value: float) -> str:
  return str(float(value))


def _run_dir(
    results_dir: Path,
    rs: str,
    d: str,
    seed: int,
) -> Path:
  return (
      results_dir
      / f"N24_layers12_12_rs{rs}_d{d}_D20.0_seed{seed}_sq"
  )


def _parse_schedule(value: str) -> list[float]:
  schedule = [float(item) for item in value.split(",") if item.strip()]
  if not schedule:
    raise ValueError("--d-schedule must contain at least one value")
  return schedule


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
  parser.add_argument("--d", type=float, default=4.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--iterations-per-stage", type=int, default=1000)
  parser.add_argument(
      "--rs-schedule",
      type=_parse_schedule,
      default=_parse_schedule("2.0,1.5,1.0,0.8,0.6,0.5"),
      help="Comma-separated r_s values, e.g. 2.0,1.5,1.0,0.8,0.6,0.5.")
  parser.add_argument(
      "--results-dir",
      type=Path,
      default=REPO_ROOT / "results" / "anneal_rs",
      help="Parent directory for stage result folders.")
  parser.add_argument(
      "--keep-optimizer-state",
      action="store_true",
      help="Reuse optimizer state between stages. By default only params, "
           "walkers, and MCMC width are reused.")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  results_dir = args.results_dir
  if not results_dir.is_absolute():
    results_dir = REPO_ROOT / results_dir

  d = _float_string(args.d)
  restore_path = ""

  for stage, rs_value in enumerate(args.rs_schedule):
    rs = _float_string(rs_value)
    stage_dir = _run_dir(results_dir, rs, d, args.seed)

    env = os.environ.copy()
    env["SCAN_RS"] = rs
    env["SCAN_D"] = d
    env["SCAN_SEED"] = str(args.seed)
    env["SCAN_ITERATIONS"] = str(args.iterations_per_stage)
    env["RESULTS_DIR"] = str(results_dir)
    env["RESET_ITERATION_ON_RESTORE"] = "1"
    env["RESET_OPTIMIZER_ON_RESTORE"] = (
        "0" if args.keep_optimizer_state else "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not old_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{old_pythonpath}")

    if restore_path:
      env["RESTORE_PATH"] = restore_path

    cmd = [sys.executable, str(args.run_script)]
    print(
        f"Stage {stage:02d}: rs={rs}, d={d}, seed={args.seed}, "
        f"save={stage_dir}",
        flush=True)
    if restore_path:
      print(f"  restoring from {restore_path}", flush=True)

    if args.dry_run:
      continue

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    restore_path = str(stage_dir)


if __name__ == "__main__":
  main()
