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


num_bosons = 32
layer_occupations = (16,16)
layer_separation = 1.0
dipole_strength = 50.0
supercell_shape = "sq"
density_rs = 0.5

burn_in_cut = 0
rolling_window = 100

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


def add_rolling_average(ax, x, y, label, color=None):
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
    rolling = y.rolling(rolling_window, min_periods=rolling_window // 5).mean()
    ax.plot(
        x,
        rolling,
        linewidth=1.8,
        color=color,
        label=f"{label}, rolling mean")


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
plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
axs = axs.ravel()
steps = plot_data["step"]

add_rolling_average(
    axs[0],
    steps,
    plot_data["energy"] / num_bosons,
    "energy / N",
    color="#2a6fbb")
if "ewmean" in plot_data:
  axs[0].plot(
      steps,
      plot_data["ewmean"] / num_bosons,
      linewidth=1.5,
      color="#111111",
      label="EW mean / N")
axs[0].set_ylabel("energy / N")
axs[0].legend(fontsize=8)

if "locstd" in plot_data:
  locstd_per_particle = plot_data["locstd"] / num_bosons
  add_rolling_average(
      axs[1],
      steps,
      locstd_per_particle,
      "std(E_L) / N",
      color="#c7364f")
  axs[1].set_ylabel("std(E_L) / N")
  if (locstd_per_particle > 0).all():
    axs[1].set_yscale("log")
  axs[1].legend(fontsize=8)
else:
  axs[1].text(0.5, 0.5, "locstd unavailable", ha="center", va="center")

if "locstd" in plot_data:
  variance_per_particle = (plot_data["locstd"] / num_bosons) ** 2
  add_rolling_average(
      axs[2],
      steps,
      variance_per_particle,
      "var(E_L / N)",
      color="#2f7d57")
  axs[2].set_ylabel("var(E_L / N)")
  if (variance_per_particle > 0).all():
    axs[2].set_yscale("log")
  axs[2].legend(fontsize=8)
elif "ewvar" in plot_data:
  ewvar_per_particle = plot_data["ewvar"] / (num_bosons ** 2)
  add_rolling_average(
      axs[2],
      steps,
      ewvar_per_particle,
      "EW var(E_L / N)",
      color="#2f7d57")
  axs[2].set_ylabel("EW var(E_L / N)")
  if (ewvar_per_particle > 0).all():
    axs[2].set_yscale("log")
  axs[2].legend(fontsize=8)
else:
  axs[2].text(0.5, 0.5, "variance unavailable", ha="center", va="center")

if "pmove" in plot_data:
  add_rolling_average(
      axs[3],
      steps,
      plot_data["pmove"],
      "pmove",
      color="#a36f00")
  axs[3].set_ylabel("MCMC acceptance")
  axs[3].set_ylim(0.0, 1.0)
  axs[3].legend(fontsize=8)
else:
  axs[3].text(0.5, 0.5, "pmove unavailable", ha="center", va="center")

for ax in axs:
  ax.set_xlabel("step")
  ax.grid(alpha=0.25, linewidth=0.5)

fig.tight_layout()
output_path = os.path.join(folder_name, "training_diagnostics.png")
fig.savefig(output_path, dpi=200)
print(f"Saved training diagnostics plot to {output_path}")
plt.close(fig)
