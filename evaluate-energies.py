# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Plot energy convergence for a bilayer boson run."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


num_bosons = 14
layer_occupations = (7, 7)
layer_separation = 10.0
dipole_strength = 20.0
density_rs = 3.0
supercell_shape = "sq"
burn_in_cut = 0

folder_name = (
    "results/bilayer-bosons/"
    f"BosonNet/N{num_bosons}_layers{layer_occupations[0]}_{layer_occupations[1]}"
    f"_rs{density_rs}_d{layer_separation}_D{dipole_strength}_{supercell_shape}"
)


def load_train_stats(folder_path):
  stats_files = ["train_stats.csv"] + sorted(
      f for f in os.listdir(folder_path)
      if f.startswith("train_stats_") and f.endswith(".csv"))
  for stats_file in stats_files:
    path = os.path.join(folder_path, stats_file)
    if not os.path.exists(path):
      continue
    data = pd.read_csv(path)
    if not data.empty:
      print(f"Loaded {path}")
      return data
  raise ValueError(f"No non-empty train_stats*.csv found in {folder_path}")


train_data = load_train_stats(folder_name)
if len(train_data) <= burn_in_cut:
  print(
      f"Only {len(train_data)} rows found; plotting all rows instead of "
      f"applying cut {burn_in_cut}.")
  burn_in_cut = 0

plot_data = train_data.iloc[burn_in_cut:]

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(
    plot_data["step"],
    plot_data["energy"] / num_bosons,
    marker="o",
    linestyle="-",
    linewidth=0.4,
    markersize=1,
    alpha=0.35,
    label="energy per boson")
if "ewmean" in plot_data:
  ax.plot(
      plot_data["step"],
      plot_data["ewmean"] / num_bosons,
      linewidth=1.6,
      label="weighted mean per boson")
ax.set_xlabel("step")
ax.set_ylabel("energy per boson")
ax.legend()
fig.tight_layout()
output_path = os.path.join(folder_name, "energy.png")
fig.savefig(output_path, dpi=200)
print(f"Saved energy plot to {output_path}")
