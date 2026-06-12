#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Run one shard of an r_s,d scan by launching run_bilayer.py."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "scan_manifests" / "rs_d_manifest.csv"
DEFAULT_RUN_SCRIPT = REPO_ROOT / "scripts" / "train" / "run_bilayer.py"


def _int_from_env(name: str, default: int) -> int:
  value = os.environ.get(name)
  return default if value is None else int(value)


def _read_manifest(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise ValueError(f"{path} does not contain any scan points")
  missing = [name for name in ("rs", "d") if name not in rows[0]]
  if missing:
    raise ValueError(f"{path} is missing required columns: {missing}")
  return rows


def _float_string(value: str) -> str:
  return str(float(value))


def _row_args(row: dict[str, str]) -> list[str]:
  args = ["--rs", _float_string(row["rs"]), "--d", _float_string(row["d"])]
  if row.get("seed"):
    args.extend(["--seed", str(int(float(row["seed"])))])
  return args


def _run_point(
    run_script: Path,
    row: dict[str, str],
    passthrough_args: list[str],
    dry_run: bool,
) -> None:
  cmd = [sys.executable, str(run_script), *passthrough_args, *_row_args(row)]
  print("command=" + " ".join(cmd), flush=True)
  if dry_run:
    return

  env = os.environ.copy()
  env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
  env.setdefault("XDG_CACHE_HOME", "/tmp")
  old_pythonpath = env.get("PYTHONPATH")
  env["PYTHONPATH"] = (
      str(REPO_ROOT) if not old_pythonpath
      else f"{REPO_ROOT}{os.pathsep}{old_pythonpath}")

  subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=__doc__,
      epilog=(
          "Arguments not consumed by this scan wrapper are forwarded to "
          "scripts/train/run_bilayer.py for every manifest row."))
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
  parser.add_argument("--rank", type=int,
                      default=_int_from_env("LLSUB_RANK", 0))
  parser.add_argument("--size", type=int,
                      default=_int_from_env("LLSUB_SIZE", 1))
  parser.add_argument("--dry-run", action="store_true",
                      help="Print commands without running them.")
  args, passthrough_args = parser.parse_known_args()

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
    _run_point(args.run_script, row, passthrough_args, args.dry_run)


if __name__ == "__main__":
  main()
