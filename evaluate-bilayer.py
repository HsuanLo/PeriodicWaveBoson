# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Evaluate density and structure factor for a bilayer boson run."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs("/tmp/matplotlib", exist_ok=True)
os.makedirs("/tmp/fontconfig", exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from periodicwave.pbc import lattices


num_bosons = 24
layer_occupations = (12, 12)
layer_separation = 2.0
dipole_strength = 20.0
supercell_shape = "sq"
density_rs = 1.0

load_N_ckpts = 1
snapshot_count = 12
pair_correlation_bins = 80

folder_name = (
    "results/bilayer-bosons/"
    f"BosonNet/N{num_bosons}_layers{layer_occupations[0]}_{layer_occupations[1]}"
    f"_rs{density_rs}_d{layer_separation}_D{dipole_strength}_{supercell_shape}"
)

layer_assignment = np.array(
    [1.0] * layer_occupations[0] + [-1.0] * layer_occupations[1])

if supercell_shape == "tri":
  supercell_a = density_rs * np.sqrt(2 * np.pi / np.sqrt(3) * num_bosons)
  lat_vec, _ = lattices._triangular_lattice_vecs_periodic_potential(
      supercell_a, 1)
elif supercell_shape == "sq":
  supercell_a = density_rs * np.sqrt(np.pi * num_bosons)
  lat_vec = lattices._square_lattice_vecs(supercell_a)
else:
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")

rec = 2 * np.pi * np.linalg.inv(lat_vec)


def latest_checkpoint_files(folder_path, nfiles):
  files = [
      f for f in os.listdir(folder_path)
      if f.startswith("qmcjax_ckpt_") and f.endswith(".npz")
  ]
  numbered = []
  for file in files:
    try:
      numbered.append((int(file.split("_")[-1].split(".")[0]), file))
    except ValueError:
      pass
  return [name for _, name in sorted(numbered, reverse=True)[:nfiles]]


def load_positions(folder_path, nfiles):
  positions = []
  ckpt_files = latest_checkpoint_files(folder_path, nfiles)
  if not ckpt_files:
    raise ValueError(f"No checkpoints found in {folder_path}")
  print(f"Loading checkpoints: {ckpt_files}")
  for filename in ckpt_files:
    ckpt = np.load(os.path.join(folder_path, filename), allow_pickle=True)
    data = ckpt["data"].item()
    pos = data["positions"]
    positions.append(pos)
  arr = np.asarray(positions)
  arr = arr.reshape((-1, num_bosons, 2))
  return np.asarray([
      lattices.send_positions_to_first_unit_cell(config, lat_vec, rec)
      for config in arr
  ])


def cell_outline():
  corners_frac = np.array([
      [-0.5, -0.5],
      [0.5, -0.5],
      [0.5, 0.5],
      [-0.5, 0.5],
      [-0.5, -0.5],
  ])
  return np.einsum("ij,kj->ki", lat_vec, corners_frac)


def minimum_image(displacements):
  frac = np.einsum("ij,...j->...i", np.linalg.inv(lat_vec), displacements)
  frac = (frac + 0.5) % 1.0 - 0.5
  return np.einsum("ij,...j->...i", lat_vec, frac)


def scatter_density(ax, positions, title, color):
  flat = positions.reshape((-1, 2))
  npoints = flat.shape[0]
  point_size = np.clip(1800 / max(npoints, 1), 1.5, 8.0)
  alpha = 0.7

  ax.scatter(
      flat[:, 0],
      flat[:, 1],
      s=point_size,
      alpha=alpha,
      color=color,
      linewidths=0,
      rasterized=True)

  outline = cell_outline()
  ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.0, alpha=0.65)
  pad = 0.04 * np.max(np.ptp(outline, axis=0))
  ax.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
  ax.set_ylim(outline[:, 1].min() - pad, outline[:, 1].max() + pad)
  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title(f"{title} ({npoints} samples)")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)


def draw_cell(ax):
  outline = cell_outline()
  ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.0, alpha=0.65)
  pad = 0.04 * np.max(np.ptp(outline, axis=0))
  ax.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
  ax.set_ylim(outline[:, 1].min() - pad, outline[:, 1].max() + pad)
  ax.set_aspect("equal", adjustable="box")
  ax.set_xticks([])
  ax.set_yticks([])


def save_snapshot_plots(positions, nsnapshots=snapshot_count):
  nsnapshots = min(nsnapshots, positions.shape[0])
  if nsnapshots == 0:
    return
  indices = np.linspace(0, positions.shape[0] - 1, nsnapshots, dtype=int)
  ncols = min(4, nsnapshots)
  nrows = int(np.ceil(nsnapshots / ncols))
  fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
  axs = np.atleast_1d(axs).ravel()
  colors = np.where(layer_assignment == 1.0, "#c7364f", "#2f7d57")
  for ax, idx in zip(axs, indices):
    config = positions[idx]
    ax.scatter(
        config[:, 0],
        config[:, 1],
        s=32,
        c=colors,
        edgecolors="black",
        linewidths=0.4)
    draw_cell(ax)
    ax.set_title(f"sample {idx}")
  for ax in axs[len(indices):]:
    ax.axis("off")
  fig.tight_layout()
  path = os.path.join(folder_name, "bilayer_snapshots.png")
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def compute_pair_correlation(positions, nbins=pair_correlation_bins):
  area = abs(np.linalg.det(lat_vec))
  density = num_bosons / area
  lengths = np.linalg.norm(lat_vec, axis=0)
  rmax = 0.5 * np.min(lengths)
  edges = np.linspace(0.0, rmax, nbins + 1)
  counts = np.zeros(nbins)
  for config in positions:
    disp = config[:, None, :] - config[None, :, :]
    disp = minimum_image(disp)
    distances = np.linalg.norm(disp, axis=-1)
    pair_distances = distances[np.triu_indices(num_bosons, k=1)]
    counts += np.histogram(pair_distances, bins=edges)[0]
  centers = 0.5 * (edges[:-1] + edges[1:])
  shell_areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
  ideal_counts = positions.shape[0] * num_bosons * density * shell_areas / 2.0
  gr = np.divide(
      counts,
      ideal_counts,
      out=np.zeros_like(counts, dtype=float),
      where=ideal_counts > 0)
  return centers, gr


def save_pair_correlation(positions):
  r, gr = compute_pair_correlation(positions)
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  ax.plot(r, gr, linewidth=1.5)
  ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
  ax.set_xlabel("r")
  ax.set_ylabel("g(r)")
  ax.set_title("Pair correlation")
  ax.grid(alpha=0.25, linewidth=0.5)
  fig.tight_layout()
  path = os.path.join(folder_name, "bilayer_pair_correlation.png")
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def save_density_plots(positions):
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  scatter_density(ax, positions, "Overall xy density", color="#2a6fbb")
  fig.tight_layout()
  path = os.path.join(folder_name, "bilayer_density_xy.png")
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)

  fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
  masks = [layer_assignment == 1.0, layer_assignment == -1.0]
  titles = ["top layer", "bottom layer"]
  colors = ["#c7364f", "#2f7d57"]
  for ax, mask, title, color in zip(axs, masks, titles, colors):
    scatter_density(ax, positions[:, mask, :], title, color)
  fig.tight_layout()
  path = os.path.join(folder_name, "bilayer_density_layers.png")
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)

  fig, ax = plt.subplots(1, 1, figsize=(5, 4))
  zs = [-0.5 * layer_separation, 0.5 * layer_separation]
  counts = [np.sum(layer_assignment == -1.0), np.sum(layer_assignment == 1.0)]
  ax.bar(zs, counts, width=0.2 * layer_separation)
  ax.set_xlabel("z")
  ax.set_ylabel("particle count")
  ax.set_title("Density along z")
  fig.tight_layout()
  path = os.path.join(folder_name, "bilayer_density_z.png")
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


def compute_structure_factor(positions, kmax=5):
  ms = []
  vals = []
  for m1 in range(-kmax, kmax + 1):
    for m2 in range(-kmax, kmax + 1):
      if m1 == 0 and m2 == 0:
        continue
      kvec = rec @ np.array([m1, m2])
      phases = np.exp(1j * np.einsum("cnd,d->cn", positions, kvec))
      rho_k = np.sum(phases, axis=1)
      sk = np.mean(np.abs(rho_k) ** 2) / num_bosons
      ms.append((m1, m2))
      vals.append(sk)
  return np.asarray(ms), np.asarray(vals)


def save_structure_factor(positions):
  ms, vals = compute_structure_factor(positions)
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  scatter = ax.scatter(ms[:, 0], ms[:, 1], c=vals, s=80, cmap="viridis")
  fig.colorbar(scatter, ax=ax, label="S(k)")
  ax.set_xlabel("m1")
  ax.set_ylabel("m2")
  ax.set_title("Static structure factor")
  ax.set_aspect("equal", adjustable="box")
  fig.tight_layout()
  path = os.path.join(folder_name, "bilayer_structure_factor.png")
  fig.savefig(path, dpi=200)
  print(f"Saved {path}")
  plt.close(fig)


positions = load_positions(folder_name, load_N_ckpts)
save_density_plots(positions)
save_snapshot_plots(positions)
save_pair_correlation(positions)
save_structure_factor(positions)
