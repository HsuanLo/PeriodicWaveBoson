# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Minimal network interfaces for the bilayer boson VMC path."""

from typing import Any, Iterable, MutableMapping, Tuple, Union

import attr
import chex
import jax.numpy as jnp
from typing_extensions import Protocol


ParamTree = Union[
    jnp.ndarray, Iterable["ParamTree"], MutableMapping[Any, "ParamTree"]
]
Param = MutableMapping[str, jnp.ndarray]


@chex.dataclass
class WalkerData:
  """Walker data container.

  `positions` are boson xy coordinates and `spins` are fixed layer labels.
  """

  positions: Any
  spins: Any
  atoms: Any
  charges: Any


class InitNetwork(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network."""


class NetworkLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      positions: jnp.ndarray,
      layer_labels: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns `(phase, log_abs)` for a walker configuration."""


class LogNetworkLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      positions: jnp.ndarray,
      layer_labels: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns `log_abs` for a walker configuration."""


class OrbitalFnLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      positions: jnp.ndarray,
      layer_labels: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the network's scalar hidden output."""


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
  """Options common to the bilayer boson network."""

  ndim: int = 2
  complex_output: bool = False
  pbc_lattice: jnp.ndarray = jnp.array([])


@attr.s(auto_attribs=True)
class Network:
  options: BaseNetworkOptions
  init: InitNetwork
  apply: NetworkLike
  orbitals: OrbitalFnLike
