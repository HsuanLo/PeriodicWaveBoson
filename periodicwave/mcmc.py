# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Metropolis-Hastings Monte Carlo for bilayer boson walkers."""

import chex
from periodicwave import constants
from periodicwave import network_interfaces as networks
import jax
from jax import lax
from jax import numpy as jnp


def _mh_update(
    params: networks.ParamTree,
    f: networks.LogNetworkLike,
    data: networks.WalkerData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    sum_esjd,
    sum_moved_particles,
    moved_particles_per_walker,
    proposal_mask=None,
    stddev=0.02,
):
  """Performs one Metropolis-Hastings move."""
  key, subkey = jax.random.split(key)
  x1 = data.positions
  layer_labels = data.spins

  displacement = stddev * jax.random.normal(subkey, shape=x1.shape)
  if proposal_mask is not None:
    displacement = jnp.where(proposal_mask, displacement, 0.0)
  x2 = x1 + displacement
  lp_2 = 2.0 * f(params, x2, layer_labels, data.atoms, data.charges)
  ratio = lp_2 - lp_1

  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  accept = ratio > rnd
  x_new = jnp.where(accept[..., None], x2, x1)
  lp_new = jnp.where(accept, lp_2, lp_1)
  num_accepts += jnp.sum(accept)
  squared_jump = jnp.sum(displacement ** 2, axis=-1)
  sum_esjd += jnp.sum(jnp.where(accept, squared_jump, 0.0))
  sum_moved_particles += jnp.sum(
      jnp.ones_like(accept) * moved_particles_per_walker)
  new_data = networks.WalkerData(
      **(dict(data) | {"positions": x_new, "spins": layer_labels}))
  return new_data, key, lp_new, num_accepts, sum_esjd, sum_moved_particles


def _block_mask(
    key: chex.PRNGKey,
    positions: jnp.ndarray,
    ndim: int,
    block_size: int,
) -> jnp.ndarray:
  """Returns a random particle block mask expanded over coordinates."""
  num_particles = positions.shape[-1] // ndim
  block_size = min(block_size, num_particles)
  scores = jax.random.uniform(key, shape=positions.shape[:-1] + (num_particles,))
  particle_ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
  particle_mask = particle_ranks < block_size
  return jnp.repeat(particle_mask, ndim, axis=-1)


def _proposal_update(
    params: networks.ParamTree,
    f: networks.LogNetworkLike,
    data: networks.WalkerData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    sum_esjd,
    sum_moved_particles,
    proposal: str,
    ndim: int,
    block_size: int,
    global_move_fraction: float,
    global_width_scale: float,
    stddev=0.02,
):
  """Dispatches one MH update for the selected proposal family."""
  num_particles = data.positions.shape[-1] // ndim
  block_size = min(block_size, num_particles)
  if proposal == "global":
    return _mh_update(
        params,
        f,
        data,
        key,
        lp_1,
        num_accepts,
        sum_esjd,
        sum_moved_particles,
        num_particles,
        stddev=stddev)

  key, proposal_key, choose_key = jax.random.split(key, 3)
  block_mask = _block_mask(proposal_key, data.positions, ndim, block_size)

  if proposal == "block":
    return _mh_update(
        params,
        f,
        data,
        key,
        lp_1,
        num_accepts,
        sum_esjd,
        sum_moved_particles,
        block_size,
        proposal_mask=block_mask,
        stddev=stddev)

  if proposal == "hybrid":
    use_global = jax.random.uniform(choose_key) < global_move_fraction
    global_stddev = stddev * global_width_scale
    return jax.lax.cond(
        use_global,
        lambda _: _mh_update(
            params,
            f,
            data,
            key,
            lp_1,
            num_accepts,
            sum_esjd,
            sum_moved_particles,
            num_particles,
            stddev=global_stddev),
        lambda _: _mh_update(
            params,
            f,
            data,
            key,
            lp_1,
            num_accepts,
            sum_esjd,
            sum_moved_particles,
            block_size,
            proposal_mask=block_mask,
            stddev=stddev),
        operand=None)

  raise ValueError(
      f"Unknown MCMC proposal {proposal!r}; expected global, block, or hybrid.")


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   ndim=2,
                   blocks=1,
                   proposal="global",
                   block_size=1,
                   global_move_fraction=0.2,
                   global_width_scale=0.25):
  """Creates the MCMC step function."""
  if blocks != 1:
    raise NotImplementedError("Bilayer boson MCMC currently uses blocks=1.")
  if proposal not in ("global", "block", "hybrid"):
    raise ValueError(
        f"Unknown MCMC proposal {proposal!r}; expected global, block, or hybrid.")
  if block_size < 1:
    raise ValueError("MCMC block_size must be at least 1.")
  if not 0.0 <= global_move_fraction <= 1.0:
    raise ValueError("MCMC global_move_fraction must be between 0 and 1.")
  if global_width_scale <= 0.0:
    raise ValueError("MCMC global_width_scale must be positive.")

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps."""
    logprob = 2.0 * batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)

    def step_fn(_, state):
      return _proposal_update(
          params,
          batch_network,
          *state,
          proposal=proposal,
          ndim=ndim,
          block_size=block_size,
          global_move_fraction=global_move_fraction,
          global_width_scale=global_width_scale,
          stddev=width)

    new_data, key, _, num_accepts, sum_esjd, sum_moved_particles = (
        lax.fori_loop(0, steps, step_fn, (data, key, logprob, 0.0, 0.0, 0.0)))
    pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
    pmove = constants.pmean(pmove)
    num_particles = data.positions.shape[-1] // ndim
    esjd_per_particle = sum_esjd / (steps * batch_per_device * num_particles)
    esjd_per_particle = constants.pmean(esjd_per_particle)
    mean_sum_esjd = constants.pmean(sum_esjd)
    mean_sum_moved_particles = constants.pmean(sum_moved_particles)
    esjd_per_moved_particle = mean_sum_esjd / mean_sum_moved_particles
    return new_data, pmove, esjd_per_particle, esjd_per_moved_particle

  return mcmc_step
