# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Benchmark bilayer dipolar Ewald against brute-force image sums."""

import argparse
import itertools

import numpy as np

from periodicwave.pbc import lattices

jax = None
jnp = None
bilayer_hamiltonian = None


def _require_runtime_deps() -> None:
  global jax, jnp, bilayer_hamiltonian
  if jax is not None:
    return
  try:
    import jax as jax_module
    import jax.numpy as jnp_module
    from periodicwave.pbc import bilayer_hamiltonian as bilayer_hamiltonian_module
  except ModuleNotFoundError as exc:
    if exc.name == "jax":
      raise SystemExit(
          "benchmark_ewald.py requires JAX. Run it in the training "
          "environment where jax and jaxlib are installed.") from exc
    if exc.name == "chex":
      raise SystemExit(
          "benchmark_ewald.py requires the project training dependencies, "
          "including chex.") from exc
    raise
  jax = jax_module
  jnp = jnp_module
  bilayer_hamiltonian = bilayer_hamiltonian_module


num_bosons = 14
layer_occupations = (7, 7)
layer_separation = 0.5
density_rs = 1.0
dipole_strength = 20.0
ewald_softening = 0.1
soften_same_layer_only = False
supercell_shape = "sq"

direct_image_cutoffs = [2, 4, 6, 8, 10, 12]
ewald_real_cuts = [2, 3, 4]
ewald_kmax_values = [8, 10, 12, 16]
ewald_alpha_multipliers = [2.0, 4.0, 6.0]


def make_lattice():
  if supercell_shape == "tri":
    supercell_a = density_rs * np.sqrt(
        2 * np.pi / np.sqrt(3) * num_bosons)
    lattice, _ = lattices._triangular_lattice_vecs_periodic_potential(
        supercell_a, 1)
    return lattice
  if supercell_shape == "sq":
    supercell_a = density_rs * np.sqrt(np.pi * num_bosons)
    return lattices._square_lattice_vecs(supercell_a)
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")


lat_vec = make_lattice()
cell_length = np.sqrt(abs(np.linalg.det(lat_vec)))
default_alpha = 5.0 / cell_length
layer_assignment = np.array(
    [1.0] * layer_occupations[0] + [-1.0] * layer_occupations[1])


def fractional_to_cartesian(frac):
  return np.einsum("ij,kj->ki", lat_vec, np.asarray(frac))


def minimum_image_xy(displacements):
  rec_no_2pi = np.linalg.inv(lat_vec)
  fractional = np.einsum("ij,...j->...i", rec_no_2pi, displacements)
  fractional = (fractional + 0.5) % 1.0 - 0.5
  return np.einsum("ij,...j->...i", lat_vec, fractional)


def dipolar_energy(rho2, dz2):
  r2 = np.maximum(rho2 + dz2, 1e-24)
  return dipole_strength * (1.0 - 3.0 * dz2 / r2) / (r2 ** 1.5)


def deterministic_configs():
  same_layer_pair = (
      fractional_to_cartesian([[0.0, 0.0], [0.18, 0.07]]),
      np.array([1.0, 1.0]),
  )
  opposite_layer_pair = (
      fractional_to_cartesian([[0.0, 0.0], [0.02, 0.01]]),
      np.array([1.0, -1.0]),
  )

  rng = np.random.default_rng(1234)
  random_frac = rng.uniform(-0.45, 0.45, size=(num_bosons, 2))
  random_config = (fractional_to_cartesian(random_frac), layer_assignment)

  stripe_frac = np.array([
      [-0.36, -0.30], [-0.24, -0.18], [-0.12, -0.06], [0.00, 0.06],
      [0.12, 0.18], [0.24, 0.30], [0.36, 0.42], [-0.36, 0.30],
      [-0.24, 0.18], [-0.12, 0.06], [0.00, -0.06], [0.12, -0.18],
      [0.24, -0.30], [0.36, -0.42],
  ])
  stripe_config = (fractional_to_cartesian(stripe_frac), layer_assignment)

  return {
      "same_layer_pair": same_layer_pair,
      "opposite_layer_pair": opposite_layer_pair,
      "random_config": random_config,
      "stripe_config": stripe_config,
  }


def brute_force_image_energy(positions, labels, image_cutoff):
  positions = np.asarray(positions)
  labels = np.asarray(labels)
  z = 0.5 * layer_separation * labels
  images = []
  for n1, n2 in itertools.product(
      range(-image_cutoff, image_cutoff + 1),
      range(-image_cutoff, image_cutoff + 1)):
    images.append(lat_vec @ np.array([n1, n2]))
  images = np.asarray(images)

  energy = 0.0
  for i in range(positions.shape[0]):
    for j in range(i + 1, positions.shape[0]):
      rho = positions[i] - positions[j] + images
      dz = z[i] - z[j]
      rho2 = np.sum(rho ** 2, axis=1)
      energy += np.sum(dipolar_energy(rho2, dz ** 2))
      if ewald_softening > 0 and (
          not soften_same_layer_only or labels[i] == labels[j]):
        rho_min = minimum_image_xy(positions[i] - positions[j])
        rho2_min = np.sum(rho_min ** 2)
        dz2 = dz ** 2
        bare = dipolar_energy(rho2_min, dz2)
        soft = dipolar_energy(rho2_min + ewald_softening ** 2, dz2)
        energy += soft - bare
  return energy


def ewald_energy(positions, labels, alpha, real_cut, kmax):
  potential = bilayer_hamiltonian.make_ewald_bilayer_potential(
      lattice=jnp.asarray(lat_vec),
      layer_separation=layer_separation,
      dipole_strength=dipole_strength,
      softening=ewald_softening,
      ewald_alpha=alpha,
      ewald_real_cut=real_cut,
      ewald_kmax=kmax,
      ewald_geometry="xy_periodic_open_z")
  return float(potential(jnp.asarray(positions.reshape(-1)), jnp.asarray(labels)))


def print_direct_convergence(name, positions, labels):
  label = "brute-force image sum"
  if ewald_softening > 0:
    label += " + local softening correction"
  print(f"\n{name}: {label}")
  print("  image_cut   energy")
  reference = None
  for cutoff in direct_image_cutoffs:
    energy = brute_force_image_energy(positions, labels, cutoff)
    reference = energy
    print(f"  {cutoff:9d}   {energy: .12e}")
  return reference


def print_ewald_scan(name, positions, labels, reference):
  print(f"\n{name}: Ewald scan")
  print("  alpha_mult real_cut kmax   energy             delta_vs_direct")
  for alpha_mult in ewald_alpha_multipliers:
    alpha = alpha_mult * default_alpha
    for real_cut in ewald_real_cuts:
      for kmax in ewald_kmax_values:
        energy = ewald_energy(positions, labels, alpha, real_cut, kmax)
        delta = energy - reference
        print(
            f"  {alpha_mult:10.3f} {real_cut:8d} {kmax:4d}   "
            f"{energy: .12e}   {delta: .4e}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.parse_args()

  _require_runtime_deps()
  print("Bilayer dipolar Ewald benchmark")
  print(f"num_bosons          = {num_bosons}")
  print(f"layer_occupations   = {layer_occupations}")
  print(f"layer_separation    = {layer_separation}")
  print(f"density_rs          = {density_rs}")
  print(f"dipole_strength     = {dipole_strength}")
  print(f"ewald_softening     = {ewald_softening}")
  softening_scope = "same_layer_only" if soften_same_layer_only else "all_pairs"
  print(f"softening_scope     = {softening_scope}")
  print(f"cell area           = {abs(np.linalg.det(lat_vec)):.12e}")
  print(f"default alpha       = {default_alpha:.12e}")
  print(f"jax backend         = {jax.default_backend()}")

  for name, (positions, labels) in deterministic_configs().items():
    reference = print_direct_convergence(name, positions, labels)
    print_ewald_scan(name, positions, labels, reference)


if __name__ == "__main__":
  main()
