# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Permutation-symmetric bosonic neural wavefunction ansatz."""

from typing import Mapping, Sequence, Tuple

import attr
import chex
from periodicwave import network_interfaces as networks
from periodicwave import network_layers
import jax
import jax.numpy as jnp


@attr.s(auto_attribs=True, kw_only=True)
class BosonNetOptions(networks.BaseNetworkOptions):
  """Options for a symmetric bosonic wavefunction network.

  The network returns a positive real wavefunction by default:
  phase = 1 and log_abs = scalar network output.
  """

  architecture: str = "DeepSets"
  num_layers: int = 3
  mlp_dim: int = 64
  num_heads: int = 4
  attn_dim: int = 16
  value_dim: int = 16
  num_perceptrons_per_layer: int = 2
  use_layer_norm: bool = True
  mlp_activation_fct: str = "GELU"
  layer_separation: float = 1.0


def _activation(name: str):
  name = name.upper()
  if name == "TANH":
    return jnp.tanh
  if name == "ELU":
    return jax.nn.elu
  if name == "GELU":
    return jax.nn.gelu
  raise ValueError(f"Unknown activation function: {name}")


def make_layer_norm():
  """Returns init/apply functions for LayerNorm."""

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    return {
        "scale": jnp.ones(param_shape),
        "offset": jnp.zeros(param_shape),
    }

  def apply(params: networks.ParamTree, inputs: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    variance = jnp.var(inputs, axis=-1, keepdims=True)
    return params["scale"] * jax.lax.rsqrt(variance + 1e-5) * (
        inputs - mean) + params["offset"]

  return init, apply


def make_mlp(activation_fct_name: str):
  """Creates a shared MLP builder."""

  activation_fct = _activation(activation_fct_name)

  def init(key: chex.PRNGKey, dims: Sequence[int]) -> Sequence[networks.Param]:
    params = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
      key, subkey = jax.random.split(key)
      params.append(
          network_layers.init_linear_layer(
              subkey, in_dim=in_dim, out_dim=out_dim, include_bias=True))
    return params

  def apply(params: Sequence[networks.Param],
            inputs: jnp.ndarray,
            activate_final: bool = True) -> jnp.ndarray:
    x = inputs
    for i, layer in enumerate(params):
      x = network_layers.linear_layer(x, **layer)
      if activate_final or i < len(params) - 1:
        x = activation_fct(x)
    return x

  return init, apply


def make_transformer_block(num_heads: int,
                           embed_dim: int,
                           attn_dim: int,
                           value_dim: int,
                           ff_dim: int,
                           activation_fct_name: str):
  """Creates a pre-norm permutation-equivariant Transformer block."""

  activation_fct = _activation(activation_fct_name)
  ln_init, ln_apply = make_layer_norm()

  def init(key: chex.PRNGKey) -> Mapping[str, jnp.ndarray]:
    key, q_key, k_key, v_key, out_key, ff1_key, ff2_key = jax.random.split(
        key, 7)
    del key
    return {
        "attn_ln": ln_init(embed_dim),
        "ff_ln": ln_init(embed_dim),
        "q_w": network_layers.init_linear_layer(
            q_key, embed_dim, num_heads * attn_dim, include_bias=False)["w"],
        "k_w": network_layers.init_linear_layer(
            k_key, embed_dim, num_heads * attn_dim, include_bias=False)["w"],
        "v_w": network_layers.init_linear_layer(
            v_key, embed_dim, num_heads * value_dim, include_bias=False)["w"],
        "out_w": network_layers.init_linear_layer(
            out_key, num_heads * value_dim, embed_dim, include_bias=False)["w"],
        "ff1": network_layers.init_linear_layer(
            ff1_key, embed_dim, ff_dim, include_bias=True),
        "ff2": network_layers.init_linear_layer(
            ff2_key, ff_dim, embed_dim, include_bias=True),
    }

  def apply(params: networks.ParamTree, tokens: jnp.ndarray) -> jnp.ndarray:
    num_tokens = tokens.shape[0]
    attn_input = ln_apply(params["attn_ln"], tokens)
    q = jnp.dot(attn_input, params["q_w"]).reshape(
        num_tokens, num_heads, attn_dim)
    k = jnp.dot(attn_input, params["k_w"]).reshape(
        num_tokens, num_heads, attn_dim)
    v = jnp.dot(attn_input, params["v_w"]).reshape(
        num_tokens, num_heads, value_dim)
    attended = jax.nn.dot_product_attention(q, k, v)
    attended = attended.reshape(num_tokens, num_heads * value_dim)
    x = tokens + network_layers.linear_layer(attended, params["out_w"])

    ff_input = ln_apply(params["ff_ln"], x)
    hidden = activation_fct(network_layers.linear_layer(
        ff_input, **params["ff1"]))
    return x + network_layers.linear_layer(hidden, **params["ff2"])

  return init, apply


def _periodic_particle_features(pos: jnp.ndarray,
                                layer_labels: jnp.ndarray,
                                lattice: jnp.ndarray,
                                ndim: int) -> jnp.ndarray:
  """Creates per-boson periodic features from xy coordinates and layer labels."""
  positions = jnp.reshape(pos, [-1, ndim])
  reciprocal_no_2pi = jnp.linalg.inv(lattice)
  scaled = jnp.einsum("ij,kj->ki", reciprocal_no_2pi, positions)
  periodic = jnp.concatenate(
      [jnp.sin(2 * jnp.pi * scaled), jnp.cos(2 * jnp.pi * scaled)], axis=-1)
  return jnp.concatenate([periodic, layer_labels[:, None]], axis=-1)


def make_boson_net(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 2,
    complex_output: bool = False,
    pbc_lattice: jnp.ndarray,
    layer_separation: float = 1.0,
    architecture: str,
    num_layers: int,
    mlp_dim: int,
    num_heads: int,
    attn_dim: int,
    value_dim: int,
    num_perceptrons_per_layer: int,
    use_layer_norm: bool,
    mlp_activation_fct: str,
) -> networks.Network:
  """Builds a permutation-symmetric bosonic wavefunction.

  `spins` in the apply function are interpreted as bilayer labels, +1 for the
  upper layer and -1 for the lower layer.
  """

  del charges, layer_separation
  if complex_output:
    raise NotImplementedError("BosonNet currently implements real positive wavefunctions only.")
  if ndim != 2:
    raise ValueError("BosonNet expects ndim=2 continuous xy coordinates.")

  architecture = architecture.upper()
  if architecture not in {"DEEPSETS", "ATTENTION", "TRANSFORMER"}:
    raise ValueError(f"Unknown BosonNet architecture: {architecture}")
  use_transformer = architecture in {"ATTENTION", "TRANSFORMER"}

  options = BosonNetOptions(
      ndim=ndim,
      complex_output=complex_output,
      pbc_lattice=pbc_lattice,
      architecture=architecture,
      num_layers=num_layers,
      mlp_dim=mlp_dim,
      num_heads=num_heads,
      attn_dim=attn_dim,
      value_dim=value_dim,
      num_perceptrons_per_layer=num_perceptrons_per_layer,
      use_layer_norm=use_layer_norm,
      mlp_activation_fct=mlp_activation_fct,
  )

  num_bosons = int(sum(nspins))
  input_dim = 2 * ndim + 1
  mlp_init, mlp_apply = make_mlp(mlp_activation_fct)
  ln_init, ln_apply = make_layer_norm()
  transformer_init, transformer_apply = make_transformer_block(
      num_heads,
      mlp_dim,
      attn_dim,
      value_dim,
      ff_dim=4 * mlp_dim,
      activation_fct_name=mlp_activation_fct)

  def init(key: chex.PRNGKey) -> networks.ParamTree:
    params = {}
    key, embed_key = jax.random.split(key)
    params["embed"] = network_layers.init_linear_layer(
        embed_key, input_dim, mlp_dim, include_bias=True)

    params["phi"] = []
    params["transformer"] = []
    params["ln"] = []
    for _ in range(num_layers):
      key, mlp_key, attn_key = jax.random.split(key, 3)
      dims = [mlp_dim] * (num_perceptrons_per_layer + 1)
      params["phi"].append(mlp_init(mlp_key, dims))
      if use_transformer:
        params["transformer"].append(transformer_init(attn_key))
      elif use_layer_norm:
        params["ln"].append(ln_init(mlp_dim))

    key, rho_key = jax.random.split(key)
    params["rho"] = mlp_init(rho_key, [mlp_dim, mlp_dim, 1])
    return params

  def hidden_apply(params: networks.ParamTree,
                   pos: jnp.ndarray,
                   spins: jnp.ndarray,
                   atoms: jnp.ndarray,
                   charges: jnp.ndarray) -> jnp.ndarray:
    del atoms, charges
    if spins.shape[0] != num_bosons:
      raise ValueError("BosonNet received the wrong number of layer labels.")
    features = _periodic_particle_features(pos, spins, pbc_lattice, ndim)
    x = network_layers.linear_layer(features, **params["embed"])
    x = _activation(mlp_activation_fct)(x)

    for layer in range(num_layers):
      if use_transformer:
        x = transformer_apply(params["transformer"][layer], x)
      else:
        x = x + mlp_apply(params["phi"][layer], x)
      if not use_transformer and use_layer_norm:
        x = ln_apply(params["ln"][layer], x)

    pooled = jnp.mean(x, axis=0)
    return mlp_apply(params["rho"], pooled, activate_final=False)[0]

  def apply(params: networks.ParamTree,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray):
    log_abs = hidden_apply(params, pos, spins, atoms, charges)
    phase = jnp.asarray(1.0)
    return phase, log_abs

  return networks.Network(
      options=options,
      init=init,
      apply=apply,
      orbitals=hidden_apply,
  )
