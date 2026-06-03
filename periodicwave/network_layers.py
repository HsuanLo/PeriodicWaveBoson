# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0.

"""Neural network building blocks used by BosonNet."""

from typing import MutableMapping, Optional

import chex
import jax
import jax.numpy as jnp


def init_linear_layer(
    key: chex.PRNGKey, in_dim: int, out_dim: int, include_bias: bool = True
) -> MutableMapping[str, jnp.ndarray]:
  """Initializes parameters for a linear layer, x w + b."""
  key1, key2 = jax.random.split(key)
  weight = (
      jax.random.normal(key1, shape=(in_dim, out_dim)) /
      jnp.sqrt(float(in_dim)))
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {"w": weight, "b": bias}
  return {"w": weight}


def linear_layer(x: jnp.ndarray,
                 w: jnp.ndarray,
                 b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Evaluates a linear layer, x w + b."""
  y = jnp.dot(x, w)
  return y + b if b is not None else y


vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)
