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

from periodicwave.configs import bilayer_bosons


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_SCRIPT = REPO_ROOT / "scripts" / "train" / "run_bilayer.py"
DEFAULTS = bilayer_bosons.DEFAULTS


def _float_string(value: float) -> str:
  return str(float(value))


def _run_dir(
    results_dir: Path,
    num_bosons: int,
    layers: str,
    rs: str,
    d: str,
    dipole_strength: str,
    seed: int,
    cell: str,
) -> Path:
  return (
      results_dir
      / f"N{num_bosons}_layers{layers}_rs{rs}_d{d}_D{dipole_strength}_seed{seed}_{cell}"
  )


def _parse_schedule(value: str) -> list[float]:
  schedule = [float(item) for item in value.split(",") if item.strip()]
  if not schedule:
    raise ValueError("--d-schedule must contain at least one value")
  return schedule


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
  parser.add_argument("--num-bosons", type=int, default=DEFAULTS.num_bosons)
  parser.add_argument(
      "--layers",
      default=bilayer_bosons.format_layer_occupations(
          DEFAULTS.layer_occupations))
  parser.add_argument("--d", type=float, default=DEFAULTS.layer_separation)
  parser.add_argument("--dipole-strength", type=float,
                      default=DEFAULTS.dipole_strength)
  parser.add_argument("--cell", default=DEFAULTS.supercell_shape)
  parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
  parser.add_argument("--burn-in-iterations", type=int,
                      default=DEFAULTS.burn_in_iterations)
  parser.add_argument("--bold-iterations", type=int,
                      default=DEFAULTS.bold_iterations)
  parser.add_argument("--retune-iterations", type=int,
                      default=DEFAULTS.retune_iterations)
  parser.add_argument("--fine-iterations", type=int,
                      default=DEFAULTS.fine_iterations)
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
      "--restore-path",
      default=os.environ.get("RESTORE_PATH", ""),
      help="Checkpoint directory used to initialize the first rs stage.")
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
  dipole_strength = _float_string(args.dipole_strength)
  restore_path = args.restore_path

  for stage, rs_value in enumerate(args.rs_schedule):
    rs = _float_string(rs_value)
    stage_dir = _run_dir(
        results_dir,
        args.num_bosons,
        args.layers,
        rs,
        d,
        dipole_strength,
        args.seed,
        args.cell)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not old_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{old_pythonpath}")
    if args.keep_optimizer_state:
      env["RESET_OPTIMIZER_ON_RESTORE"] = "0"

    cmd = [
        sys.executable,
        str(args.run_script),
        "--num-bosons",
        str(args.num_bosons),
        "--layers",
        args.layers,
        "--rs",
        rs,
        "--d",
        d,
        "--dipole-strength",
        dipole_strength,
        "--cell",
        args.cell,
        "--seed",
        str(args.seed),
        "--burn-in-iterations",
        str(args.burn_in_iterations),
        "--bold-iterations",
        str(args.bold_iterations),
        "--retune-iterations",
        str(args.retune_iterations),
        "--fine-iterations",
        str(args.fine_iterations),
        "--results-dir",
        str(results_dir),
        "--reset-iteration-on-restore",
    ]
    if not args.keep_optimizer_state:
      cmd.append("--reset-optimizer-on-restore")
    if restore_path:
      cmd.extend(["--restore-path", restore_path])

    print(
        f"Stage {stage:02d}: rs={rs}, d={d}, seed={args.seed}, "
        f"save={stage_dir}",
        flush=True)
    if restore_path:
      print(f"  restoring from {restore_path}", flush=True)

    if args.dry_run:
      restore_path = str(stage_dir)
      continue

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    restore_path = str(stage_dir)


if __name__ == "__main__":
  main()
