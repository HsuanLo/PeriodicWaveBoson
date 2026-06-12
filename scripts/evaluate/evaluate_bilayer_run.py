#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Run the default bilayer evaluation workflow for scan runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_energies
import evaluate_observables


DEFAULT_SCAN_DIR = evaluate_energies.DEFAULT_SCAN_DIR
DEFAULT_PATTERN = evaluate_energies.DEFAULT_PATTERN
VALID_STAGES = ("training", "observables")


def _parse_stages(value: str) -> tuple[str, ...]:
  if value == "all":
    return VALID_STAGES
  stages = tuple(stage.strip() for stage in value.split(",") if stage.strip())
  unknown = sorted(set(stages) - set(VALID_STAGES))
  if unknown:
    raise argparse.ArgumentTypeError(
        f"unknown stage(s): {', '.join(unknown)}; use training, observables, or all")
  if not stages:
    raise argparse.ArgumentTypeError("at least one stage is required")
  return stages


def _evaluate_run(
    run_dir: Path,
    stages: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
  if "training" in stages:
    evaluate_energies._evaluate_run(  # pylint: disable=protected-access
        run_dir,
        args.burn_in_cut,
        args.rolling_window,
        args.skip_existing,
        args.write_extra_figures,
    )

  if "observables" in stages:
    evaluate_observables._evaluate_run(  # pylint: disable=protected-access
        run_dir,
        args.load_n_ckpts,
        args.max_configs,
        args.snapshot_count,
        args.pair_correlation_bins,
        args.kmax,
        args.skip_existing,
        args.write_extra_figures,
    )


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
      "--stages",
      type=_parse_stages,
      default=VALID_STAGES,
      help="Comma-separated stages to run: training, observables, or all.",
  )
  parser.add_argument("--burn-in-cut", type=int, default=0)
  parser.add_argument("--rolling-window", type=int, default=100)
  parser.add_argument("--load-n-ckpts", type=int, default=1)
  parser.add_argument(
      "--max-configs",
      type=int,
      default=0,
      help="Subsample configurations per run; use 0 for all.",
  )
  parser.add_argument("--snapshot-count", type=int, default=12)
  parser.add_argument("--pair-correlation-bins", type=int, default=80)
  parser.add_argument("--kmax", type=int, default=5)
  parser.add_argument(
      "--skip-existing",
      action="store_true",
      help="Skip stages whose default outputs already exist.",
  )
  parser.add_argument(
      "--write-extra-figures",
      action="store_true",
      help=(
          "Also write demoted figures: standalone energy trace, separate "
          "density/S(k), snapshots, z-density, and total g(r)."
      ),
  )
  args = parser.parse_args()

  args.max_configs = None if args.max_configs == 0 else args.max_configs
  run_dirs = evaluate_energies._select_run_dirs(  # pylint: disable=protected-access
      args.run_dir, args.scan_dir, args.pattern)

  failures = []
  for idx, run_dir in enumerate(run_dirs, start=1):
    print(f"Evaluating run {idx}/{len(run_dirs)}: {run_dir.name}", flush=True)
    try:
      _evaluate_run(run_dir, args.stages, args)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failures.append((run_dir, exc))
      print(f"Failed {run_dir.name}: {exc}", flush=True)

  print(f"Processed {len(run_dirs) - len(failures)}/{len(run_dirs)} runs")
  if failures:
    print(f"Failures: {len(failures)}")
    for path, exc in failures[:10]:
      print(f"  {path.name}: {exc}")


if __name__ == "__main__":
  main()
