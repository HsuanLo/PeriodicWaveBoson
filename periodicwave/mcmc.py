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
    stddev=0.02,
):
  """Performs one all-particle Metropolis-Hastings move."""
  key, subkey = jax.random.split(key)
  x1 = data.positions
  layer_labels = data.spins

  x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)
  lp_2 = 2.0 * f(params, x2, layer_labels, data.atoms, data.charges)
  ratio = lp_2 - lp_1

  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  accept = ratio > rnd
  x_new = jnp.where(accept[..., None], x2, x1)
  lp_new = jnp.where(accept, lp_2, lp_1)
  num_accepts += jnp.sum(accept)
  new_data = networks.WalkerData(
      **(dict(data) | {"positions": x_new, "spins": layer_labels}))
  return new_data, key, lp_new, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   ndim=2,
                   blocks=1):
  """Creates the MCMC step function."""
  if blocks != 1:
    raise NotImplementedError("Bilayer boson MCMC currently uses blocks=1.")
  del ndim

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps."""
    logprob = 2.0 * batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)

    def step_fn(_, state):
      return _mh_update(
          params,
          batch_network,
          *state,
          stddev=width)

    new_data, key, _, num_accepts = lax.fori_loop(
        0, steps, step_fn, (data, key, logprob, 0.0))
    pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step
