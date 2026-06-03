#!/usr/bin/env python3
# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Estimate bilayer one-body density matrices from BosonNet checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from periodicwave.pbc import lattices

jax = None
jnp = None
BosonNet = None


def _require_jax() -> None:
  global jax, jnp, BosonNet
  if jax is not None:
    return
  try:
    import jax as jax_module
    import jax.numpy as jnp_module
    from periodicwave import BosonNet as boson_net_module
  except ModuleNotFoundError as exc:
    if exc.name == "jax":
      raise SystemExit(
          "evaluate_one_body_density_matrix.py requires JAX because it "
          "evaluates the checkpointed BosonNet wavefunction. Run it in the "
          "training environment where jax and jaxlib are installed.") from exc
    raise
  jax = jax_module
  jnp = jnp_module
  BosonNet = boson_net_module


DEFAULT_SCAN_DIR = (
    REPO_ROOT
    / "results"
    / "bilayer-bosons"
    / "BosonNet"
    / "scan_260603"
)
DEFAULT_PATTERN = "N24_layers12_12_rs*_d*_D20.0_sq"
RUN_RE = re.compile(
    r"N(?P<num_bosons>\d+)_layers(?P<layer_a>\d+)_(?P<layer_b>\d+)"
    r"_rs(?P<rs>[0-9.]+)_d(?P<d>[0-9.]+)_D(?P<dipole>[0-9.]+)_"
    r"(?P<cell>[^/]+)$"
)


@dataclass(frozen=True)
class RunParams:
  path: Path
  num_bosons: int
  layer_occupations: tuple[int, int]
  rs: float
  d: float
  dipole_strength: float
  supercell_shape: str
  lat_vec: np.ndarray
  rec: np.ndarray
  layer_assignment: np.ndarray
  network_options: dict[str, Any]


@dataclass(frozen=True)
class CheckpointData:
  step: int
  params: Any
  positions: np.ndarray
  spins: np.ndarray
  atoms: np.ndarray
  charges: np.ndarray


@dataclass(frozen=True)
class MomentumBasis:
  modes: np.ndarray
  k_vectors: np.ndarray


@dataclass(frozen=True)
class MomentumEstimate:
  gamma: np.ndarray
  occupations: np.ndarray
  eigenvalues: np.ndarray
  eigenvectors: np.ndarray


def _lattice(num_bosons: int, rs: float, supercell_shape: str) -> np.ndarray:
  if supercell_shape == "tri":
    supercell_a = rs * np.sqrt(2 * np.pi / np.sqrt(3) * num_bosons)
    lat_vec, _ = lattices._triangular_lattice_vecs_periodic_potential(
        supercell_a, 1)
    return lat_vec
  if supercell_shape == "sq":
    supercell_a = rs * np.sqrt(np.pi * num_bosons)
    return lattices._square_lattice_vecs(supercell_a)
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")


def _read_config(run_dir: Path) -> dict[str, Any]:
  config_path = run_dir / "config.json"
  if not config_path.exists():
    return {}
  with config_path.open(encoding="utf-8") as f:
    return json.load(f)


def _parse_run_dir(run_dir: Path) -> RunParams:
  match = RUN_RE.match(run_dir.name)
  if not match:
    raise ValueError(f"Cannot parse parameters from {run_dir.name}")
  parsed = match.groupdict()
  config = _read_config(run_dir)
  num_bosons = int(parsed["num_bosons"])
  layer_occupations = (int(parsed["layer_a"]), int(parsed["layer_b"]))
  rs = float(parsed["rs"])
  d = float(parsed["d"])
  supercell_shape = parsed["cell"]
  lat_vec = _lattice(num_bosons, rs, supercell_shape)
  rec = 2 * np.pi * np.linalg.inv(lat_vec)
  layer_assignment = np.array(
      [1.0] * layer_occupations[0] + [-1.0] * layer_occupations[1])
  network_options = dict(config.get("network", {}).get("BosonNet", {}))
  if not network_options:
    network_options = {
        "architecture": "Transformer",
        "num_layers": 3,
        "mlp_dim": 64,
        "num_heads": 4,
        "attn_dim": 16,
        "value_dim": 16,
        "num_perceptrons_per_layer": 2,
        "use_layer_norm": True,
        "mlp_activation_fct": "GELU",
    }
  return RunParams(
      path=run_dir,
      num_bosons=num_bosons,
      layer_occupations=layer_occupations,
      rs=rs,
      d=d,
      dipole_strength=float(parsed["dipole"]),
      supercell_shape=supercell_shape,
      lat_vec=lat_vec,
      rec=rec,
      layer_assignment=layer_assignment,
      network_options=network_options,
  )


def _latest_checkpoint(run_dir: Path) -> Path:
  numbered = []
  for path in run_dir.glob("qmcjax_ckpt_*.npz"):
    try:
      numbered.append((int(path.stem.split("_")[-1]), path))
    except ValueError:
      pass
  if not numbered:
    raise ValueError(f"No qmcjax_ckpt_*.npz files found in {run_dir}")
  return sorted(numbered, reverse=True)[0][1]


def _unreplicate_tree(tree):
  def maybe_first_device(x):
    arr = np.asarray(x)
    if arr.ndim > 0 and arr.shape[0] == jax.local_device_count():
      return arr[0]
    if arr.ndim > 0 and arr.shape[0] == jax.device_count():
      return arr[0]
    if arr.ndim > 0 and arr.shape[0] == 1:
      return arr[0]
    return arr

  return jax.tree_util.tree_map(maybe_first_device, tree)


def _flatten_leading_axes(value: np.ndarray) -> np.ndarray:
  arr = np.asarray(value)
  if arr.ndim >= 3:
    return arr.reshape((-1,) + arr.shape[2:])
  if arr.ndim == 2:
    return arr.reshape((-1,))
  return arr


def _load_checkpoint(run_dir: Path) -> CheckpointData:
  ckpt_path = _latest_checkpoint(run_dir)
  ckpt = np.load(ckpt_path, allow_pickle=True)
  data = ckpt["data"].item()
  params = _unreplicate_tree(ckpt["params"].tolist())
  positions = _flatten_leading_axes(np.asarray(data["positions"]))
  spins = _flatten_leading_axes(np.asarray(data["spins"]))
  atoms = _flatten_leading_axes(np.asarray(data["atoms"]))
  charges = _flatten_leading_axes(np.asarray(data["charges"]))
  return CheckpointData(
      step=int(ckpt["t"].tolist()),
      params=params,
      positions=positions,
      spins=spins,
      atoms=atoms,
      charges=charges,
  )


def _subsample_configs(ckpt: CheckpointData, max_configs: int, seed: int) -> CheckpointData:
  if max_configs <= 0 or ckpt.positions.shape[0] <= max_configs:
    return ckpt
  rng = np.random.default_rng(seed)
  indices = np.sort(rng.choice(ckpt.positions.shape[0], size=max_configs, replace=False))
  return CheckpointData(
      step=ckpt.step,
      params=ckpt.params,
      positions=ckpt.positions[indices],
      spins=ckpt.spins[indices],
      atoms=ckpt.atoms[indices],
      charges=ckpt.charges[indices],
  )


def _wrap_positions(positions: jnp.ndarray, lattice: jnp.ndarray) -> jnp.ndarray:
  xy = positions.reshape((positions.shape[0], -1, 2))
  rec_no_2pi = jnp.linalg.inv(lattice)
  frac = jnp.einsum("ij,bnj->bni", rec_no_2pi, xy)
  frac = (frac + 0.5) % 1.0 - 0.5
  wrapped = jnp.einsum("ij,bnj->bni", lattice, frac)
  return wrapped.reshape(positions.shape)


def _chunked_logabs(
    logabs_batch,
    params,
    positions: np.ndarray,
    spins: np.ndarray,
    atoms: np.ndarray,
    charges: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
  values = []
  for start in range(0, positions.shape[0], chunk_size):
    end = min(start + chunk_size, positions.shape[0])
    values.append(np.asarray(logabs_batch(
        params,
        jnp.asarray(positions[start:end]),
        jnp.asarray(spins[start:end]),
        jnp.asarray(atoms[start:end]),
        jnp.asarray(charges[start:end]),
    )))
  return np.concatenate(values, axis=0)


def _build_network(params: RunParams):
  charges = jnp.asarray([0.0])
  return BosonNet.make_boson_net(
      (params.num_bosons, 0),
      charges,
      ndim=2,
      complex_output=False,
      pbc_lattice=jnp.asarray(params.lat_vec),
      layer_separation=params.d,
      **params.network_options,
  )


def _directions(num_directions: int) -> np.ndarray:
  angles = np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False)
  return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _proposal_positions(
    positions: np.ndarray,
    particle_indices: np.ndarray,
    displacements: np.ndarray,
    lat_vec: np.ndarray,
) -> np.ndarray:
  nconfig = positions.shape[0]
  proposals = np.repeat(positions[:, None, None, :], len(displacements), axis=1)
  proposals = np.repeat(proposals, len(particle_indices), axis=2)
  proposals = proposals.reshape((nconfig * len(displacements) * len(particle_indices), -1))
  proposal_particles = np.tile(particle_indices, nconfig * len(displacements))
  proposal_displacements = np.repeat(displacements, len(particle_indices), axis=0)
  proposal_displacements = np.tile(proposal_displacements, (nconfig, 1))
  xy = proposals.reshape((-1, positions.shape[1] // 2, 2))
  xy[np.arange(xy.shape[0]), proposal_particles] += proposal_displacements

  rec_no_2pi = np.linalg.inv(lat_vec)
  frac = np.einsum("ij,bnj->bni", rec_no_2pi, xy)
  frac = (frac + 0.5) % 1.0 - 0.5
  wrapped = np.einsum("ij,bnj->bni", lat_vec, frac)
  return wrapped.reshape(proposals.shape)


def _proposal_positions_absolute(
    positions: np.ndarray,
    particle_indices: np.ndarray,
    replacement_positions: np.ndarray,
    lat_vec: np.ndarray,
) -> np.ndarray:
  nconfig = positions.shape[0]
  nreplace = replacement_positions.shape[0]
  nparticle = len(particle_indices)
  proposals = np.repeat(positions[:, None, None, :], nreplace, axis=1)
  proposals = np.repeat(proposals, nparticle, axis=2)
  proposals = proposals.reshape((nconfig * nreplace * nparticle, -1))
  proposal_particles = np.tile(particle_indices, nconfig * nreplace)
  proposal_replacements = np.repeat(replacement_positions, nparticle, axis=0)
  proposal_replacements = np.tile(proposal_replacements, (nconfig, 1))
  xy = proposals.reshape((-1, positions.shape[1] // 2, 2))
  xy[np.arange(xy.shape[0]), proposal_particles] = proposal_replacements

  rec_no_2pi = np.linalg.inv(lat_vec)
  frac = np.einsum("ij,bnj->bni", rec_no_2pi, xy)
  frac = (frac + 0.5) % 1.0 - 0.5
  wrapped = np.einsum("ij,bnj->bni", lat_vec, frac)
  return wrapped.reshape(proposals.shape)


def _estimate_layer_rho1(
    logabs_batch,
    params: RunParams,
    ckpt: CheckpointData,
    old_logabs: np.ndarray,
    radii: np.ndarray,
    direction_vectors: np.ndarray,
    particle_indices: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  means = []
  stderrs = []
  atoms = np.repeat(ckpt.atoms[:, None, ...],
                    len(direction_vectors) * len(particle_indices), axis=1)
  atoms = atoms.reshape((-1,) + ckpt.atoms.shape[1:])
  charges = np.repeat(ckpt.charges[:, None, ...],
                      len(direction_vectors) * len(particle_indices), axis=1)
  charges = charges.reshape((-1,) + ckpt.charges.shape[1:])
  spins = np.repeat(ckpt.spins[:, None, :],
                    len(direction_vectors) * len(particle_indices), axis=1)
  spins = spins.reshape((-1, ckpt.spins.shape[-1]))
  old = np.repeat(
      old_logabs[:, None],
      len(direction_vectors) * len(particle_indices),
      axis=1,
  ).reshape(-1)

  for radius in radii:
    displacements = radius * direction_vectors
    proposed = _proposal_positions(
        ckpt.positions, particle_indices, displacements, params.lat_vec)
    new_logabs = _chunked_logabs(
        logabs_batch,
        ckpt.params,
        proposed,
        spins,
        atoms,
        charges,
        chunk_size,
    )
    ratios = np.exp(np.clip(new_logabs - old, -60.0, 60.0))
    means.append(float(np.mean(ratios)))
    stderrs.append(float(np.std(ratios, ddof=1) / np.sqrt(ratios.size)))
  return np.asarray(means), np.asarray(stderrs)


def _build_momentum_basis(params: RunParams, kmax: int) -> MomentumBasis:
  modes = np.array(
      [(m1, m2) for m1 in range(-kmax, kmax + 1)
       for m2 in range(-kmax, kmax + 1)],
      dtype=int,
  )
  # With frac = inv(lat_vec) @ r, k = 2 pi * m @ inv(lat_vec).
  k_vectors = modes @ params.rec
  return MomentumBasis(modes=modes, k_vectors=k_vectors)


def _sample_replacement_positions(
    lat_vec: np.ndarray,
    num_replacement_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
  frac = rng.uniform(-0.5, 0.5, size=(num_replacement_points, 2))
  return np.einsum("ij,sj->si", lat_vec, frac)


def _estimate_layer_momentum_gamma(
    logabs_batch,
    params: RunParams,
    ckpt: CheckpointData,
    old_logabs: np.ndarray,
    basis: MomentumBasis,
    replacement_positions: np.ndarray,
    particle_indices: np.ndarray,
    layer_size: int,
    chunk_size: int,
) -> MomentumEstimate:
  if len(particle_indices) == 0:
    raise ValueError("Cannot estimate momentum 1RDM with no selected particles.")

  nconfig = ckpt.positions.shape[0]
  nreplace = replacement_positions.shape[0]
  nparticle = len(particle_indices)
  proposed = _proposal_positions_absolute(
      ckpt.positions, particle_indices, replacement_positions, params.lat_vec)
  atoms = np.repeat(ckpt.atoms[:, None, ...], nreplace * nparticle, axis=1)
  atoms = atoms.reshape((-1,) + ckpt.atoms.shape[1:])
  charges = np.repeat(ckpt.charges[:, None, ...], nreplace * nparticle, axis=1)
  charges = charges.reshape((-1,) + ckpt.charges.shape[1:])
  spins = np.repeat(ckpt.spins[:, None, :], nreplace * nparticle, axis=1)
  spins = spins.reshape((-1, ckpt.spins.shape[-1]))
  old = np.repeat(old_logabs[:, None], nreplace * nparticle, axis=1).reshape(-1)

  new_logabs = _chunked_logabs(
      logabs_batch,
      ckpt.params,
      proposed,
      spins,
      atoms,
      charges,
      chunk_size,
  )
  ratios = np.exp(np.clip(new_logabs - old, -60.0, 60.0))
  ratios = ratios.reshape((nconfig, nreplace, nparticle))

  original_xy = ckpt.positions.reshape((nconfig, -1, 2))[:, particle_indices, :]
  left_phase = np.exp(-1j * replacement_positions @ basis.k_vectors.T)
  right_phase = np.exp(1j * np.einsum("cpi,ki->cpk", original_xy, basis.k_vectors))
  gamma = np.einsum("csp,sa,cpb->ab", ratios, left_phase, right_phase)
  gamma *= layer_size / (nparticle * nconfig * nreplace)
  gamma = 0.5 * (gamma + gamma.conj().T)

  eigenvalues, eigenvectors = np.linalg.eigh(gamma)
  order = np.argsort(eigenvalues)[::-1]
  eigenvalues = np.real(eigenvalues[order])
  eigenvectors = eigenvectors[:, order]
  occupations = np.real(np.diag(gamma))
  return MomentumEstimate(
      gamma=gamma,
      occupations=occupations,
      eigenvalues=eigenvalues,
      eigenvectors=eigenvectors,
  )


def _write_csv(
    output_path: Path,
    radii: np.ndarray,
    top: np.ndarray,
    top_err: np.ndarray,
    bottom: np.ndarray,
    bottom_err: np.ndarray,
) -> None:
  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["r", "rho1_top", "rho1_top_stderr", "rho1_bottom", "rho1_bottom_stderr"])
    for row in zip(radii, top, top_err, bottom, bottom_err):
      writer.writerow([f"{value:.12g}" for value in row])


def _write_momentum_matrix_csv(
    output_path: Path,
    basis: MomentumBasis,
    gamma: np.ndarray,
) -> None:
  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "row_m1", "row_m2", "row_kx", "row_ky",
        "col_m1", "col_m2", "col_kx", "col_ky",
        "gamma_real", "gamma_imag",
    ])
    for row_idx, row_mode in enumerate(basis.modes):
      for col_idx, col_mode in enumerate(basis.modes):
        row = [
            row_mode[0],
            row_mode[1],
            f"{basis.k_vectors[row_idx, 0]:.12g}",
            f"{basis.k_vectors[row_idx, 1]:.12g}",
            col_mode[0],
            col_mode[1],
            f"{basis.k_vectors[col_idx, 0]:.12g}",
            f"{basis.k_vectors[col_idx, 1]:.12g}",
            f"{gamma[row_idx, col_idx].real:.12g}",
            f"{gamma[row_idx, col_idx].imag:.12g}",
        ]
        writer.writerow(row)


def _write_momentum_occupations_csv(
    output_path: Path,
    basis: MomentumBasis,
    layer_estimates: dict[str, MomentumEstimate],
) -> None:
  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "m1", "m2", "kx", "ky", "occupation"])
    for layer, estimate in layer_estimates.items():
      for mode, k_vec, occupation in zip(
          basis.modes, basis.k_vectors, estimate.occupations):
        writer.writerow([
            layer,
            mode[0],
            mode[1],
            f"{k_vec[0]:.12g}",
            f"{k_vec[1]:.12g}",
            f"{occupation:.12g}",
        ])


def _write_momentum_eigenvalues_csv(
    output_path: Path,
    layer_estimates: dict[str, MomentumEstimate],
    layer_sizes: dict[str, int],
) -> None:
  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "eigen_index", "occupation", "fraction"])
    for layer, estimate in layer_estimates.items():
      for idx, occupation in enumerate(estimate.eigenvalues):
        writer.writerow([
            layer,
            idx,
            f"{occupation:.12g}",
            f"{occupation / layer_sizes[layer]:.12g}",
        ])


def _plot_rho1(output_path: Path, radii, top, top_err, bottom, bottom_err, params: RunParams) -> None:
  fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
  ax.errorbar(radii, top, yerr=top_err, marker="o", markersize=3, linewidth=1.2, label="top layer")
  ax.errorbar(
      radii,
      bottom,
      yerr=bottom_err,
      marker="s",
      markersize=3,
      linewidth=1.2,
      label="bottom layer")
  ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
  ax.set_xlabel("r")
  ax.set_ylabel("rho1(r)")
  ax.set_title(f"1RDM: rs={params.rs:g}, d={params.d:g}, D={params.dipole_strength:g}")
  ax.grid(alpha=0.25, linewidth=0.5)
  ax.legend()
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _plot_momentum_summary(
    output_path: Path,
    basis: MomentumBasis,
    layer_estimates: dict[str, MomentumEstimate],
    layer_sizes: dict[str, int],
    params: RunParams,
) -> None:
  k_norms = np.linalg.norm(basis.k_vectors, axis=1)
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

  for layer, estimate in layer_estimates.items():
    order = np.argsort(k_norms)
    trace_fraction = float(np.sum(estimate.occupations) / layer_sizes[layer])
    label = f"{layer} (trace/N={trace_fraction:.3f})"
    axes[0].plot(
        k_norms[order],
        estimate.occupations[order],
        marker="o",
        markersize=3,
        linewidth=1.0,
        label=label,
    )
    axes[1].plot(
        np.arange(len(estimate.eigenvalues)),
        estimate.eigenvalues / layer_sizes[layer],
        marker="o",
        markersize=3,
        linewidth=1.0,
        label=label,
    )

  axes[0].set_xlabel("|k|")
  axes[0].set_ylabel("n(k)")
  axes[0].set_title("Momentum occupation")
  axes[0].grid(alpha=0.25, linewidth=0.5)
  axes[0].legend()
  axes[1].set_xlabel("natural orbital index")
  axes[1].set_ylabel("occupation / N_layer")
  axes[1].set_title("1RDM eigenvalue fractions")
  axes[1].grid(alpha=0.25, linewidth=0.5)
  axes[1].legend()
  fig.suptitle(
      f"Momentum 1RDM: rs={params.rs:g}, d={params.d:g}, "
      f"D={params.dipole_strength:g}")
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _occupation_grid(
    basis: MomentumBasis,
    occupations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  m1_values = np.unique(basis.modes[:, 0])
  m2_values = np.unique(basis.modes[:, 1])
  grid = np.full((len(m2_values), len(m1_values)), np.nan)
  m1_index = {value: idx for idx, value in enumerate(m1_values)}
  m2_index = {value: idx for idx, value in enumerate(m2_values)}
  for mode, occupation in zip(basis.modes, occupations):
    grid[m2_index[mode[1]], m1_index[mode[0]]] = occupation
  return m1_values, m2_values, grid


def _plot_momentum_occupation_maps(
    output_path: Path,
    basis: MomentumBasis,
    layer_estimates: dict[str, MomentumEstimate],
    layer_sizes: dict[str, int],
    params: RunParams,
) -> None:
  fig, axes = plt.subplots(1, len(layer_estimates), figsize=(5.2 * len(layer_estimates), 4.6))
  if len(layer_estimates) == 1:
    axes = [axes]

  vmax = max(float(np.nanmax(estimate.occupations))
             for estimate in layer_estimates.values())
  for ax, (layer, estimate) in zip(axes, layer_estimates.items()):
    m1_values, m2_values, grid = _occupation_grid(basis, estimate.occupations)
    trace_fraction = float(np.nansum(grid) / layer_sizes[layer])
    image = ax.imshow(
        grid,
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
        extent=[
            m1_values[0] - 0.5,
            m1_values[-1] + 0.5,
            m2_values[0] - 0.5,
            m2_values[-1] + 0.5,
        ],
    )
    ax.set_title(f"{layer} n(k), trace/N={trace_fraction:.3f}")
    ax.set_xlabel("m1")
    ax.set_ylabel("m2")
    ax.set_xticks(m1_values)
    ax.set_yticks(m2_values)
    ax.axhline(0.5, color="white", linewidth=0.5, alpha=0.35)
    ax.axhline(-0.5, color="white", linewidth=0.5, alpha=0.35)
    ax.axvline(0.5, color="white", linewidth=0.5, alpha=0.35)
    ax.axvline(-0.5, color="white", linewidth=0.5, alpha=0.35)
    fig.colorbar(image, ax=ax, shrink=0.88, label="n(k)")

  fig.suptitle(
      f"Momentum occupation maps: rs={params.rs:g}, d={params.d:g}, "
      f"D={params.dipole_strength:g}")
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def _select_particle_indices(
    layer_indices: np.ndarray,
    particles_per_layer: int,
    rng: np.random.Generator,
) -> np.ndarray:
  if particles_per_layer <= 0 or particles_per_layer >= len(layer_indices):
    return layer_indices
  return np.sort(rng.choice(layer_indices, size=particles_per_layer, replace=False))


def _evaluate_run(
    run_dir: Path,
    max_configs: int,
    kmax: int,
    num_replacement_points: int,
    particles_per_layer: int,
    chunk_size: int,
    seed: int,
    skip_existing: bool,
) -> None:
  output_top = run_dir / "one_body_density_matrix_momentum_top.csv"
  output_bottom = run_dir / "one_body_density_matrix_momentum_bottom.csv"
  output_occupations = run_dir / "one_body_density_matrix_momentum_occupations.csv"
  output_eigenvalues = run_dir / "one_body_density_matrix_momentum_eigenvalues.csv"
  output_png = run_dir / "one_body_density_matrix_momentum.png"
  output_nk_map_png = run_dir / "one_body_density_matrix_momentum_nk_map.png"
  outputs = [
      output_top,
      output_bottom,
      output_occupations,
      output_eigenvalues,
      output_png,
      output_nk_map_png,
  ]
  if skip_existing and all(path.exists() for path in outputs):
    print(f"Skipping existing 1RDM: {run_dir}")
    return

  params = _parse_run_dir(run_dir)
  ckpt = _subsample_configs(_load_checkpoint(run_dir), max_configs, seed)
  network = _build_network(params)

  def single_logabs(net_params, pos, spins, atoms, charges):
    return network.apply(net_params, pos, spins, atoms, charges)[1]

  logabs_batch = jax.jit(jax.vmap(
      single_logabs, in_axes=(None, 0, 0, 0, 0), out_axes=0))

  wrapped_positions = np.asarray(_wrap_positions(
      jnp.asarray(ckpt.positions), jnp.asarray(params.lat_vec)))
  ckpt = CheckpointData(
      step=ckpt.step,
      params=ckpt.params,
      positions=wrapped_positions,
      spins=ckpt.spins,
      atoms=ckpt.atoms,
      charges=ckpt.charges,
  )
  old_logabs = _chunked_logabs(
      logabs_batch,
      ckpt.params,
      ckpt.positions,
      ckpt.spins,
      ckpt.atoms,
      ckpt.charges,
      chunk_size,
  )

  rng = np.random.default_rng(seed)
  basis = _build_momentum_basis(params, kmax)
  replacement_positions = _sample_replacement_positions(
      params.lat_vec, num_replacement_points, rng)
  all_top_indices = np.flatnonzero(params.layer_assignment == 1.0)
  all_bottom_indices = np.flatnonzero(params.layer_assignment == -1.0)
  top_indices = _select_particle_indices(
      all_top_indices, particles_per_layer, rng)
  bottom_indices = _select_particle_indices(
      all_bottom_indices, particles_per_layer, rng)

  print(
      f"1RDM {run_dir.name}: configs={ckpt.positions.shape[0]}, "
      f"k_modes={len(basis.modes)}, replacement_points={num_replacement_points}, "
      f"top_particles={len(top_indices)}, bottom_particles={len(bottom_indices)}",
      flush=True)
  top = _estimate_layer_momentum_gamma(
      logabs_batch,
      params,
      ckpt,
      old_logabs,
      basis,
      replacement_positions,
      top_indices,
      len(all_top_indices),
      chunk_size,
  )
  bottom = _estimate_layer_momentum_gamma(
      logabs_batch,
      params,
      ckpt,
      old_logabs,
      basis,
      replacement_positions,
      bottom_indices,
      len(all_bottom_indices),
      chunk_size,
  )
  layer_estimates = {"top": top, "bottom": bottom}
  layer_sizes = {"top": len(all_top_indices), "bottom": len(all_bottom_indices)}
  _write_momentum_matrix_csv(output_top, basis, top.gamma)
  _write_momentum_matrix_csv(output_bottom, basis, bottom.gamma)
  _write_momentum_occupations_csv(output_occupations, basis, layer_estimates)
  _write_momentum_eigenvalues_csv(output_eigenvalues, layer_estimates, layer_sizes)
  _plot_momentum_summary(output_png, basis, layer_estimates, layer_sizes, params)
  _plot_momentum_occupation_maps(
      output_nk_map_png, basis, layer_estimates, layer_sizes, params)
  for path in outputs:
    print(f"Saved {path}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, default=None)
  parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
  parser.add_argument("--pattern", default=DEFAULT_PATTERN)
  parser.add_argument("--max-configs", type=int, default=512)
  parser.add_argument(
      "--kmax",
      type=int,
      default=3,
      help="Use reciprocal modes -kmax..kmax along each lattice direction.",
  )
  parser.add_argument(
      "--num-replacement-points",
      type=int,
      default=64,
      help="Uniform absolute particle replacement points sampled in the cell.",
  )
  parser.add_argument(
      "--particles-per-layer",
      type=int,
      default=0,
      help="Use this many particles per layer; use 0 for all particles.",
  )
  parser.add_argument("--chunk-size", type=int, default=2048)
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument("--skip-existing", action="store_true")
  args = parser.parse_args()

  _require_jax()

  if args.run_dir is not None:
    run_dirs = [args.run_dir.resolve()]
  else:
    scan_dir = args.scan_dir.resolve()
    run_dirs = sorted(path for path in scan_dir.glob(args.pattern) if path.is_dir())
  if not run_dirs:
    raise ValueError("No run directories selected.")

  failures = []
  for idx, run_dir in enumerate(run_dirs, start=1):
    print(f"Evaluating 1RDM {idx}/{len(run_dirs)}: {run_dir.name}", flush=True)
    try:
      _evaluate_run(
          run_dir,
          args.max_configs,
          args.kmax,
          args.num_replacement_points,
          args.particles_per_layer,
          args.chunk_size,
          args.seed,
          args.skip_existing,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failures.append((run_dir, exc))
      print(f"Failed {run_dir.name}: {exc}", flush=True)

  print(f"Processed {len(run_dirs) - len(failures)}/{len(run_dirs)} 1RDM runs")
  if failures:
    print(f"Failures: {len(failures)}")
    for path, exc in failures[:10]:
      print(f"  {path.name}: {exc}")


if __name__ == "__main__":
  main()
