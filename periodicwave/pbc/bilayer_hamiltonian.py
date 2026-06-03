# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Bilayer boson Hamiltonian with xy PBC and discrete z layers."""

from typing import Optional, Sequence, Tuple

import chex
from periodicwave import hamiltonians
from periodicwave import network_interfaces as networks
import jax
import jax.numpy as jnp
from jax.scipy import special


def _minimum_image_xy(displacements: jnp.ndarray,
                      lattice: jnp.ndarray) -> jnp.ndarray:
  """Folds xy displacement vectors into the first periodic cell."""
  rec_no_2pi = jnp.linalg.inv(lattice)
  fractional = jnp.einsum("ij,...j->...i", rec_no_2pi, displacements)
  fractional = (fractional + 0.5) % 1.0 - 0.5
  return jnp.einsum("ij,...j->...i", lattice, fractional)


def make_direct_bilayer_potential(
    lattice: jnp.ndarray,
    layer_separation: float,
    dipole_strength: float = 1.0,
    softening: float = 1e-6,
):
  """Creates a direct minimum-image dipolar potential.

  The layer labels are expected to be +1 and -1. The continuous coordinates are
  xy only; z is reconstructed as z_i = 0.5 * layer_separation * layer_label_i.
  """

  def potential(positions: jnp.ndarray, layer_labels: jnp.ndarray) -> jnp.ndarray:
    xy = jnp.reshape(positions, [-1, 2])
    num_particles = xy.shape[0]
    dxy = xy[:, None, :] - xy[None, :, :]
    dxy = _minimum_image_xy(dxy, lattice)
    rho2 = jnp.sum(dxy ** 2, axis=-1)

    z = 0.5 * layer_separation * layer_labels
    dz = z[:, None] - z[None, :]
    r2 = rho2 + dz ** 2 + softening ** 2
    r = jnp.sqrt(r2)

    cos2 = dz ** 2 / r2
    dipolar = dipole_strength * (1.0 - 3.0 * cos2) / (r ** 3)

    mask = jnp.triu(jnp.ones((num_particles, num_particles), dtype=bool), k=1)
    return jnp.sum(jnp.where(mask, dipolar, 0.0))

  return potential


def _integer_grid(cutoff: int, include_zero: bool = True) -> jnp.ndarray:
  vals = jnp.arange(-cutoff, cutoff + 1)
  grid = jnp.stack(jnp.meshgrid(vals, vals, indexing="ij"), axis=-1).reshape(-1, 2)
  if include_zero:
    return grid
  return grid[jnp.any(grid != 0, axis=1)]


def make_ewald_bilayer_potential(
    lattice: jnp.ndarray,
    layer_separation: float,
    dipole_strength: float = 1.0,
    softening: float = 0.0,
    softening_floor: float = 1e-24,
    ewald_alpha: Optional[float] = None,
    ewald_real_cut: int = 4,
    ewald_kmax: int = 8,
    ewald_geometry: str = "xy_periodic_open_z",
):
  """Creates a 2D-periodic/open-z Ewald dipolar potential.

  The implemented kernel is for z-polarized dipoles in a geometry periodic in
  xy and open in z. It starts from the standard 2D-periodic Coulomb Ewald pair
  kernel and applies `-d^2/dz^2`, using

      (1 - 3 z^2 / r^2) / r^3 = -d^2(1 / r) / dz^2.

  `ewald_real_cut` and `ewald_kmax` are integer cutoffs over direct and
  reciprocal lattice image indices. The G=0 term follows the open-z 2D-periodic
  convention encoded in `_coulomb_pair`.

  If `softening > 0`, the zero real-space image is split off before summing:
  the smooth Ewald remainder is evaluated separately, and the local singular
  piece is replaced by a softened direct dipole plus a regular Ewald remainder.
  This avoids subtracting two very large same-layer collision terms.
  """

  if ewald_geometry != "xy_periodic_open_z":
    raise NotImplementedError(
        f"Unknown dipolar Ewald geometry: {ewald_geometry}")

  lattice = jnp.asarray(lattice)
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  area = jnp.abs(jnp.linalg.det(lattice))
  alpha = (
      jnp.asarray(ewald_alpha)
      if ewald_alpha is not None else 5.0 / jnp.sqrt(area))
  sqrt_pi = jnp.sqrt(jnp.pi)

  real_indices = _integer_grid(ewald_real_cut, include_zero=True)
  nonzero_real_indices = real_indices[jnp.any(real_indices != 0, axis=1)]
  nonzero_real_images = jnp.einsum(
      "ij,kj->ki", lattice, nonzero_real_indices)
  reciprocal_indices = _integer_grid(ewald_kmax, include_zero=False)
  g_vectors = jnp.einsum("ij,kj->ki", rec, reciprocal_indices)
  g_norms = jnp.linalg.norm(g_vectors, axis=1)

  def _exp_erfc_product(exp_arg: jnp.ndarray,
                        erfc_arg: jnp.ndarray) -> jnp.ndarray:
    """Evaluates exp(exp_arg) * erfc(erfc_arg) without inf * 0 NaNs."""
    x = erfc_arg
    x2 = x ** 2
    # For large positive x, erfc(x) = exp(-x^2) erfcx(x).  Use the asymptotic
    # expansion of erfcx to avoid overflow in exp(exp_arg) and underflow in
    # erfc(x).  The direct branch is clipped because jnp.where evaluates both
    # branches before selecting.
    asymptotic = (
        jnp.exp(exp_arg - x2) /
        (jnp.sqrt(jnp.pi) * x) *
        (1.0 - 0.5 / x2 + 0.75 / (x2 ** 2)))
    direct = jnp.exp(jnp.minimum(exp_arg, 80.0)) * special.erfc(x)
    return jnp.where(x > 8.0, asymptotic, direct)

  def _coulomb_pair(rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    r = jnp.sqrt(jnp.sum(rho ** 2) + dz ** 2)
    real_zero_part = special.erfc(alpha * r) / r
    return real_zero_part + _coulomb_pair_without_real_zero(rho, dz)

  def _coulomb_pair_without_real_zero(
      rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    shifted = rho[None, :] + nonzero_real_images
    r = jnp.sqrt(jnp.sum(shifted ** 2, axis=1) + dz ** 2)
    real_part = jnp.sum(special.erfc(alpha * r) / r)

    g_dot_rho = jnp.dot(g_vectors, rho)
    gz = g_norms * dz
    g_over_2a = g_norms / (2.0 * alpha)
    reciprocal_kernel = (
        _exp_erfc_product(gz, g_over_2a + alpha * dz) +
        _exp_erfc_product(-gz, g_over_2a - alpha * dz))
    reciprocal_part = (
        jnp.pi / area *
        jnp.sum(jnp.cos(g_dot_rho) * reciprocal_kernel / g_norms))

    zero_part = (
        -2.0 * jnp.pi / area *
        (dz * special.erf(alpha * dz) +
         jnp.exp(-(alpha * dz) ** 2) / (alpha * sqrt_pi)))
    return real_part + reciprocal_part + zero_part

  d2_dz2 = jax.grad(jax.grad(_coulomb_pair, argnums=1), argnums=1)
  d2_smooth_dz2 = jax.grad(
      jax.grad(_coulomb_pair_without_real_zero, argnums=1), argnums=1)

  def _dipole_pair(rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    return -dipole_strength * d2_dz2(rho, dz)

  def _smooth_dipole_pair(rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    return -dipole_strength * d2_smooth_dz2(rho, dz)

  pair_dipole = jax.vmap(_dipole_pair, in_axes=(0, 0))
  pair_smooth_dipole = jax.vmap(_smooth_dipole_pair, in_axes=(0, 0))

  def _stable_zero_image_replacement(
      rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    rho2 = jnp.sum(rho ** 2, axis=-1)
    dz2 = dz ** 2
    r2_bare = jnp.maximum(rho2 + dz2, softening_floor)
    r_bare = jnp.sqrt(r2_bare)
    bare_numerator = rho2 - 2.0 * dz2
    exp_part = jnp.exp(-(alpha * r_bare) ** 2)
    gaussian_prefactor = 2.0 * alpha / sqrt_pi

    zero_regular = (
        -special.erf(alpha * r_bare) * bare_numerator / (r_bare ** 5) +
        gaussian_prefactor * exp_part * bare_numerator / (r_bare ** 4) -
        2.0 * gaussian_prefactor * alpha ** 2 * exp_part * dz2 /
        (r_bare ** 2))

    small_same_layer = (dz2 == 0.0) & (alpha * r_bare < 1e-2)
    small_same_layer_series = (
        -4.0 * alpha ** 3 / (3.0 * sqrt_pi) +
        4.0 * alpha ** 5 * rho2 / (5.0 * sqrt_pi) -
        2.0 * alpha ** 7 * rho2 ** 2 / (7.0 * sqrt_pi))
    zero_regular = jnp.where(
        small_same_layer, small_same_layer_series, zero_regular)

    r2_soft = rho2 + dz2 + softening ** 2
    soft_direct = (1.0 - 3.0 * dz2 / r2_soft) / (r2_soft ** 1.5)
    return dipole_strength * (zero_regular + soft_direct)

  def potential(positions: jnp.ndarray, layer_labels: jnp.ndarray) -> jnp.ndarray:
    xy = jnp.reshape(positions, [-1, 2])
    num_particles = xy.shape[0]
    pair_i, pair_j = jnp.triu_indices(num_particles, k=1)
    dxy = xy[pair_i, :] - xy[pair_j, :]
    dxy = _minimum_image_xy(dxy, lattice)
    z = 0.5 * layer_separation * layer_labels
    dz = z[pair_i] - z[pair_j]
    if softening > 0:
      energy = jnp.sum(pair_smooth_dipole(dxy, dz))
      energy += jnp.sum(_stable_zero_image_replacement(dxy, dz))
    else:
      energy = jnp.sum(pair_dipole(dxy, dz))
    return energy

  return potential


def local_energy(
    f: networks.NetworkLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    lattice: Optional[jnp.ndarray] = None,
    layer_separation: float = 1.0,
    potential_type: str = "Dipolar",
    potential_kwargs: Optional[dict] = None,
    kinetic_kwargs: Optional[dict] = None,
) -> hamiltonians.LocalEnergy:
  """Creates a local energy for bilayer bosons.

  `data.spins` is interpreted as fixed layer labels (+1 upper, -1 lower).
  The kinetic energy differentiates only with respect to continuous xy
  coordinates.
  """

  del charges, nspins
  if lattice is None:
    raise ValueError("bilayer_hamiltonian.local_energy requires lattice.")
  if complex_output:
    raise NotImplementedError("Bilayer boson Hamiltonian currently expects real BosonNet output.")

  potential_kwargs = dict(potential_kwargs or {})
  kinetic_kwargs = dict(kinetic_kwargs or {})
  use_ewald = potential_kwargs.pop("use_ewald", False)
  ewald_kwargs = {
      "ewald_alpha": potential_kwargs.pop("ewald_alpha", None),
      "ewald_real_cut": potential_kwargs.pop("ewald_real_cut", 4),
      "ewald_kmax": potential_kwargs.pop("ewald_kmax", 8),
      "ewald_geometry": potential_kwargs.pop(
          "ewald_geometry", "xy_periodic_open_z"),
  }
  potential_kwargs.pop("ewald_truncation", None)

  if potential_type != "Dipolar":
    raise NotImplementedError(f"Unknown bilayer potential_type: {potential_type}")

  laplacian_method = kinetic_kwargs.get("laplacian_method", "folx")
  kinetic = hamiltonians.local_kinetic_energy(
      f,
      use_scan=use_scan,
      complex_output=complex_output,
      laplacian_method=laplacian_method)
  if use_ewald:
    potential = make_ewald_bilayer_potential(
        lattice=jnp.asarray(lattice),
        layer_separation=layer_separation,
        **potential_kwargs,
        **ewald_kwargs)
  else:
    potential = make_direct_bilayer_potential(
        lattice=jnp.asarray(lattice),
        layer_separation=layer_separation,
        **potential_kwargs)

  def _e_l(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.WalkerData,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    del key
    return kinetic(params, data) + potential(data.positions, data.spins), None

  return _e_l
