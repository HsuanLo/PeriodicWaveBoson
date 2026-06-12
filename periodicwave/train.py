# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file has been modified from the original DeepMind version.
# Changes:
# - Streamlined for materials simulations
# - Streamlined to the bilayer boson simulation path

"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import os
import sys
import time
from typing import Optional, Mapping, Sequence, Tuple, Union

from absl import logging
import chex
from periodicwave import checkpoint
from periodicwave import constants
from periodicwave import kfac_tags
from periodicwave import hamiltonians
from periodicwave import loss as qmc_loss_functions
from periodicwave import mcmc
from periodicwave import network_interfaces as networks
from periodicwave import BosonNet
from periodicwave.utils import statistics
from periodicwave.utils import writers
from periodicwave.utils import jax_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np


def _install_jax_device_put_compat():
  """Install temporary compatibility shims for newer JAX releases."""
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

  try:
    jax.device_put_replicated
  except AttributeError:
    def device_put_replicated(x, devices):
      mesh = Mesh(np.array(devices), ('x',))
      sharding = NamedSharding(mesh, P('x'))
      return jax.tree_util.tree_map(
          lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x)

    jax.device_put_replicated = device_put_replicated

  try:
    jax.device_put_sharded
  except AttributeError:
    def device_put_sharded(shards, devices):
      mesh = Mesh(np.array(devices), ('x',))
      sharding = NamedSharding(mesh, P('x'))
      return jax.tree_util.tree_map(
          lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)

    jax.device_put_sharded = device_put_sharded


_install_jax_device_put_compat()

import kfac_jax
import ml_collections
import optax
from typing_extensions import Protocol

try:
  from tqdm.auto import tqdm
except ImportError:
  tqdm = None


def _progress_bar(iterable, **kwargs):
  """Wraps an iterable with tqdm when available."""
  if tqdm is None:
    return iterable
  disable_env = os.environ.get("PW_DISABLE_TQDM")
  if disable_env is None:
    disable = jax.process_index() != 0 or not sys.stderr.isatty()
  else:
    disable = disable_env.lower() in ("1", "true", "yes", "y", "on")
  return tqdm(
      iterable,
      dynamic_ncols=True,
      disable=disable,
      **kwargs)

def _assign_layer_configuration(
    ntop: int, nbottom: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns fixed bilayer labels for the initial walkers."""
  layers = jnp.concatenate((jnp.ones(ntop), -jnp.ones(nbottom)))
  return jnp.tile(layers[None], reps=(batch_size, 1))

def init_bosons_gaussian(
    key,
    layer_occupations: Sequence[int],
    ndim: int,
    batch_size: int,
    init_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes boson positions.

  Args:
    key: JAX RNG state.
    layer_occupations: tuple of top- and bottom-layer boson counts.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of Gaussian used to generate initial configurations.

  Returns:
    Boson positions and fixed layer labels for the initial MCMC configurations.
  """

  key, subkey = jax.random.split(key)
  boson_positions = (
      jax.random.normal(
          subkey, shape=(batch_size, ndim * sum(layer_occupations))) *
      init_width)
  layer_labels = _assign_layer_configuration(
      layer_occupations[0], layer_occupations[1], batch_size)

  return boson_positions, layer_labels


def _jittered_layer_positions(
    num_particles: int,
    lattice: np.ndarray,
    offset_fractional: tuple[float, float],
) -> np.ndarray:
  """Returns one layer of approximately evenly spaced positions in the cell."""
  if num_particles <= 0:
    return np.zeros((0, 2), dtype=np.float32)
  nx = int(np.ceil(np.sqrt(num_particles)))
  ny = int(np.ceil(num_particles / nx))
  xs = (np.arange(nx, dtype=np.float32) + 0.5) / nx
  ys = (np.arange(ny, dtype=np.float32) + 0.5) / ny
  frac = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
  frac = frac[:num_particles]
  frac = (frac + np.asarray(offset_fractional, dtype=np.float32)) % 1.0
  centered_frac = frac - 0.5
  return centered_frac @ np.asarray(lattice, dtype=np.float32).T


def init_bosons_jittered_lattice(
    key,
    layer_occupations: Sequence[int],
    ndim: int,
    batch_size: int,
    lattice: np.ndarray,
    jitter_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes bilayer bosons on cell-spanning grids with small jitter."""
  if ndim != 2:
    raise ValueError("Jittered lattice initialization currently expects ndim=2.")
  layer_occupations = tuple(int(n) for n in layer_occupations)
  top_base = _jittered_layer_positions(
      layer_occupations[0], lattice, offset_fractional=(0.0, 0.0))
  bottom_nx = int(np.ceil(np.sqrt(max(layer_occupations[1], 1))))
  bottom_ny = int(np.ceil(max(layer_occupations[1], 1) / bottom_nx))
  bottom_base = _jittered_layer_positions(
      layer_occupations[1],
      lattice,
      offset_fractional=(0.5 / bottom_nx, 0.5 / bottom_ny))
  base_positions = np.concatenate([top_base, bottom_base], axis=0)
  base_positions = jnp.asarray(base_positions.reshape(-1), dtype=jnp.float32)
  base_positions = jnp.tile(base_positions[None, :], reps=(batch_size, 1))

  key, subkey = jax.random.split(key)
  jitter = (
      jax.random.normal(subkey, shape=base_positions.shape) *
      jnp.asarray(jitter_width, dtype=base_positions.dtype))
  boson_positions = base_positions + jitter
  layer_labels = _assign_layer_configuration(
      layer_occupations[0], layer_occupations[1], batch_size)
  return boson_positions, layer_labels

# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]

class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.WalkerData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: boson positions, layer labels and placeholder atom data.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.WalkerData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.WalkerData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove,
      esjd_per_particle, esjd_per_moved_particle).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
        esjd_per_particle: expected squared jumping distance per particle.
        esjd_per_moved_particle: expected squared jumping distance per proposed
          moved particle.
    """


def null_update(
    params: networks.ParamTree,
    data: networks.WalkerData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Define function that differentiates wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.WalkerData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.WalkerData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2)) # applies pmap to step with args 0,1,2 donated for memory efficiency
  def step(
      data: networks.WalkerData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove, esjd_per_particle, esjd_per_moved_particle = mcmc_step(
        params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return (
        data,
        new_params,
        new_state,
        loss,
        aux_data,
        pmove,
        esjd_per_particle,
        esjd_per_moved_particle)

  return step


def make_kfac_training_step(
    mcmc_step,
    damping: float,
    optimizer: kfac_jax.Optimizer,
    reset_if_nan: bool = False) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))
  # Due to some KFAC cleverness related to donated buffers, need to do this
  # to make state resettable
  copy_tree = constants.pmap(
      functools.partial(jax.tree_util.tree_map,
                        lambda x: (1.0 * x).astype(x.dtype)))

  def step(
      data: networks.WalkerData,
      params: networks.ParamTree,
      state: kfac_jax.Optimizer.State,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove, esjd_per_particle, esjd_per_moved_particle = mcmc_step(
        params, data, mcmc_keys, mcmc_width)

    if reset_if_nan:
      old_params = copy_tree(params)
      old_state = copy_tree(state)

    # Optimization step
    new_params, new_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        momentum=shared_mom,
        damping=shared_damping,
    )

    if reset_if_nan and jnp.any(jnp.isnan(stats['loss'])):
      new_params = old_params
      new_state = old_state
    return (
        data,
        new_params,
        new_state,
        stats['loss'],
        stats['aux'],
        pmove,
        esjd_per_particle,
        esjd_per_moved_particle)

  return step


def _best_checkpoint_score(metric: str,
                           loss: jnp.ndarray,
                           local_std: jnp.ndarray,
                           weighted_stats,
                           num_particles: int,
                           std_weight: float) -> float:
  """Returns the scalar score used to rank best checkpoint candidates."""
  if metric == 'ewmean':
    return float(np.asarray(weighted_stats.mean))
  if metric == 'energy':
    return float(np.asarray(loss))
  if metric == 'variance':
    return float(np.asarray(local_std ** 2))
  if metric == 'energy_std':
    return float(np.asarray((loss + std_weight * local_std) / num_particles))
  if metric == 'ewmean_std':
    return float(
        np.asarray((weighted_stats.mean + std_weight * local_std)
                   / num_particles))
  raise ValueError(f'Unknown best checkpoint metric: {metric}')


def _adaptive_mcmc_width_update(
    cfg: ml_collections.ConfigDict,
    mcmc_width: jnp.ndarray,
    pmove: jnp.ndarray,
    stage: Optional[ml_collections.ConfigDict] = None,
) -> jnp.ndarray:
  """Updates the MCMC proposal width from the observed acceptance rate."""
  stage = stage or {}
  target = stage.get('target_acceptance', cfg.mcmc.get('target_acceptance', 0.5))
  adapt_rate = stage.get('adapt_rate', cfg.mcmc.get('adapt_rate', 0.05))
  min_width = stage.get('min_move_width', cfg.mcmc.get('min_move_width', 1.0e-4))
  max_width = stage.get('max_move_width', cfg.mcmc.get('max_move_width', 2.0))
  log_w = jnp.log(mcmc_width) + adapt_rate * (pmove - target)
  return jnp.clip(jnp.exp(log_w), min_width, max_width)


def _stage_optimizer(stage: ml_collections.ConfigDict) -> str:
  """Returns the optimizer name for a training stage."""
  return stage.get('optimizer', 'kfac')


def _stage_iterations(stage: ml_collections.ConfigDict) -> int:
  """Returns the integer number of iterations for a stage."""
  return int(stage.get('iterations', 0))


def _configured_stages(cfg: ml_collections.ConfigDict) -> list:
  """Returns configured training stages, if present."""
  training_cfg = cfg.get('training', {})
  return list(training_cfg.get('stages', []))


def _burn_in_stage_index(cfg: ml_collections.ConfigDict):
  """Returns the configured burn-in stage index, or None for legacy configs."""
  stages = _configured_stages(cfg)
  for index, stage in enumerate(stages):
    if stage.get('name', '') == 'burn_in':
      return index
  if stages and _stage_optimizer(stages[0]) == 'none':
    return 0
  return None


def _burn_in_stage(cfg: ml_collections.ConfigDict):
  """Returns the configured burn-in stage, or None for legacy configs."""
  burn_in_index = _burn_in_stage_index(cfg)
  if burn_in_index is None:
    return None
  stages = _configured_stages(cfg)
  return stages[burn_in_index]


def _post_burn_in_stages(cfg: ml_collections.ConfigDict) -> list:
  """Returns positive-length stages after the initial burn-in stage."""
  burn_in_index = _burn_in_stage_index(cfg)
  return [
      stage for index, stage in enumerate(_configured_stages(cfg))
      if index != burn_in_index and _stage_iterations(stage) > 0
  ]


def _optimizer_stages(cfg: ml_collections.ConfigDict) -> list:
  """Returns configured optimizer stages with positive length."""
  return [
      stage for stage in _post_burn_in_stages(cfg)
      if _stage_optimizer(stage) != 'none'
  ]


def _stage_adapts_width(
    cfg: ml_collections.ConfigDict,
    stage: Optional[ml_collections.ConfigDict] = None,
) -> bool:
  """Returns whether adaptive MCMC width updates are enabled."""
  if stage is not None and 'adapt_width' in stage:
    return bool(stage.get('adapt_width'))
  return cfg.mcmc.move_width_updater == 'adaptive'


def train(cfg: ml_collections.ConfigDict, writer_manager=None, layer_assignment=None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      default_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  logging.info('Starting QMC with %i XLA devices per host '
               'across %i hosts.', num_devices, num_hosts)
  if cfg.batch_size % (num_devices * num_hosts) != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     f'got batch size {cfg.batch_size} for '
                     f'{num_devices * num_hosts} devices.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # The bilayer Hamiltonian does not use atoms or charges, but the shared network
  # call signature carries these placeholders.
  atoms = jnp.stack([jnp.array([0.0, 0.0])])
  charges = jnp.array([0.0])
  nspins = cfg.system.bosons

  # Generate atomic configurations for each walker
  batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
  batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
  batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
  batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

  if cfg.debug.deterministic:
    seed = cfg.debug.seed
  else:
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
  prng_key = jax.random.PRNGKey(seed)

  use_complex = cfg.network.get('complex', False)
  if cfg.network.network_type != 'BosonNet':
    raise ValueError('This codebase now only supports network_type="BosonNet".')
  network = BosonNet.make_boson_net(
      nspins,
      charges,
      ndim=cfg.system.ndim,
      complex_output=use_complex,
      pbc_lattice=cfg.system.pbc_lattice,
      layer_separation=cfg.system.make_local_energy_kwargs.get(
          "layer_separation", 1.0),
      **cfg.network.BosonNet,
  )
  prng_key, subkey = jax.random.split(prng_key)
  params = network.init(subkey)

  params = kfac_jax.utils.replicate_all_local_devices(params)
  signed_network = network.apply
  # Often just need log|psi(x)|.
  logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(
      logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
  )  # batched network

  # Exclusively when computing the gradient wrt the energy for complex
  # wavefunctions, it is necessary to have log(psi) rather than log(|psi|).
  # This is unused if the wavefunction is real-valued.
  def log_network(*args, **kwargs):
    if not use_complex:
      raise ValueError('This function should never be used if the '
                        'wavefunction is real-valued.')
    phase, mag = signed_network(*args, **kwargs)
    return mag + 1.j * phase

  # Set up checkpointing and restore params/data if necessary.
  # Checkpoints are saved to save_path.
  # When restoring, we first check for a checkpoint in save_path. If none are
  # found, then we check in restore_path.  This enables calculations to be
  # started from a previous calculation but then resume from their own
  # checkpoints in the event of pre-emption.
  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (
      checkpoint.find_last_checkpoint(ckpt_save_path) or
      checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    (t_init,
     data,
     params,
     opt_state_ckpt,
     mcmc_width_ckpt) = checkpoint.restore(
         ckpt_restore_filename, host_batch_size)
    if cfg.log.get('reset_iteration_on_restore', False):
      logging.info('Resetting iteration counter after checkpoint restore.')
      t_init = 0
    if cfg.optim.get('reset_optimizer_on_restore', False):
      logging.info('Resetting optimizer state after checkpoint restore.')
      opt_state_ckpt = None
  else:
    logging.info('No checkpoint found. Training new model.')
    prng_key, subkey = jax.random.split(prng_key)
    # make sure data on each host is initialized differently
    subkey = jax.random.fold_in(subkey, jax.process_index())
    if layer_assignment is None:
      layer_assignment = np.ones(sum(cfg.system.bosons))
    layer_assignment = jnp.asarray(layer_assignment)
    if layer_assignment.shape[0] != sum(cfg.system.bosons):
      raise ValueError("layer_assignment must have one entry per boson.")
    init_layer_occupations = (
        int(np.count_nonzero(np.asarray(layer_assignment) == 1.0)),
        int(np.count_nonzero(np.asarray(layer_assignment) == -1.0)))
    if sum(init_layer_occupations) != sum(cfg.system.bosons):
      raise ValueError("layer_assignment entries must be +1 or -1.")
    # Create boson positions and layer labels.
    pos, spins = init_bosons_jittered_lattice(
      subkey,
      layer_occupations=init_layer_occupations,
      ndim = cfg.system.ndim,
      batch_size=host_batch_size,
      lattice=cfg.system.pbc_lattice,
      jitter_width=cfg.mcmc.init_width,)
    spins = jnp.tile(layer_assignment[None], reps=(host_batch_size, 1))

    pos = jnp.reshape(pos, data_shape + (-1,))
    pos = kfac_jax.utils.broadcast_all_local_devices(pos)
    spins = jnp.reshape(spins, data_shape + (-1,))
    spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    data = networks.WalkerData(
        positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
    )

    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None

  burn_stage = _burn_in_stage(cfg)
  training_stages = _post_burn_in_stages(cfg)
  optimizer_stages = _optimizer_stages(cfg)
  if training_stages:
    cfg.optim.iterations = sum(_stage_iterations(stage)
                              for stage in training_stages)

  # Set up logging and observables
  train_schema = [
      'step', 'stage', 'energy', 'ewmean', 'ewvar', 'pmove', 'locstd',
      'mcmc_width', 'mcmc_esjd_per_particle',
      'mcmc_esjd_per_moved_particle'
  ]

  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(prng_key)

  # Main training

  def make_stage_mcmc_step(stage=None):
    stage = stage or {}
    return mcmc.make_mcmc_step(
        batch_network,
        device_batch_size,
        steps=stage.get('mcmc_steps', cfg.mcmc.steps),
        ndim=cfg.system.ndim,
        blocks=cfg.mcmc.blocks,
        proposal=stage.get('proposal', cfg.mcmc.get('proposal', 'global')),
        block_size=stage.get('block_size', cfg.mcmc.get('block_size', 1)),
        global_move_fraction=stage.get(
            'global_move_fraction',
            cfg.mcmc.get('global_move_fraction', 0.2)),
        global_width_scale=stage.get(
            'global_width_scale',
            cfg.mcmc.get('global_width_scale', 0.25)),
    )

  # Construct loss and optimizer
  # Requires a local energy function to be specified.
  local_energy_module, local_energy_fn = (
      cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
  local_energy_module = importlib.import_module(local_energy_module)
  make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonians.MakeLocalEnergy
  local_energy_fn = make_local_energy(
      f=signed_network,
      charges=charges,
      nspins=nspins,
      complex_output=use_complex,
      use_scan=False,
      **cfg.system.make_local_energy_kwargs)

  local_energy = local_energy_fn

  evaluate_loss = qmc_loss_functions.make_loss(
      log_network if use_complex else logabs_network,
      local_energy,
      clip_local_energy=cfg.optim.clip_local_energy,
      clip_from_median=cfg.optim.clip_median,
      center_at_clipped_energy=cfg.optim.center_at_clip,
      complex_output=use_complex,
  )

  # Compute the learning rate
  if optimizer_stages:
    stage_lengths = np.asarray([
        _stage_iterations(stage) for stage in optimizer_stages
    ], dtype=np.int32)
    stage_ends = np.cumsum(stage_lengths)
    stage_rates = np.asarray([
        stage.get('lr_rate', cfg.optim.lr.rate) for stage in optimizer_stages
    ], dtype=np.float32)

    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
      stage_idx = jnp.sum(t_ >= jnp.asarray(stage_ends[:-1]))
      return jnp.asarray(stage_rates)[stage_idx]
  else:
    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
      return cfg.optim.lr.rate * jnp.power(
          (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)

  active_optimizer_name = cfg.optim.optimizer
  if optimizer_stages:
    stage_optimizers = {_stage_optimizer(stage) for stage in optimizer_stages}
    if len(stage_optimizers) != 1:
      raise ValueError(
          'All non-burn-in training stages must use the same optimizer.')
    active_optimizer_name = next(iter(stage_optimizers))

  def kfac_learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    if active_optimizer_name != 'adam_kfac':
      return learning_rate_schedule(t_)
    return cfg.optim.adam_kfac.kfac_lr_rate * jnp.power(
        (1.0 / (1.0 + (t_/cfg.optim.adam_kfac.kfac_lr_delay))),
        cfg.optim.adam_kfac.kfac_lr_decay)

  def make_adam_optimizer() -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))

  def make_lamb_optimizer() -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))

  def make_kfac_optimizer() -> kfac_jax.Optimizer:
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    return kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=kfac_learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=kfac_tags.GRAPH_PATTERNS,
        ),
        # debug=True
    )

  def make_step_for_optimizer(optimizer, mcmc_step):
    if not optimizer:
      return make_training_step(
          mcmc_step=mcmc_step,
          optimizer_step=make_loss_step(evaluate_loss))
    if isinstance(optimizer, optax.GradientTransformation):
      return make_training_step(
          mcmc_step=mcmc_step,
          optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
          reset_if_nan=cfg.optim.reset_if_nan)
    if isinstance(optimizer, kfac_jax.Optimizer):
      return make_kfac_training_step(
          mcmc_step=mcmc_step,
          damping=cfg.optim.kfac.damping,
          optimizer=optimizer,
          reset_if_nan=cfg.optim.reset_if_nan)
    raise ValueError(f'Unknown optimizer: {optimizer}')

  def make_step_for_stage(stage):
    stage_mcmc_step = make_stage_mcmc_step(stage)
    if stage is not None and _stage_optimizer(stage) == 'none':
      if not stage.get('evaluate_loss', True):
        return make_training_step(
            mcmc_step=stage_mcmc_step,
            optimizer_step=null_update)
      return make_step_for_optimizer(None, stage_mcmc_step)
    return make_step_for_optimizer(optimizer, stage_mcmc_step)

  scheduled_adam_kfac = (
      not training_stages and active_optimizer_name == 'adam_kfac')
  switch_iteration = cfg.optim.adam_kfac.switch_iteration
  if scheduled_adam_kfac and switch_iteration <= 0:
    raise ValueError("cfg.optim.adam_kfac.switch_iteration must be positive.")

  # Construct and setup optimizer
  if active_optimizer_name == 'none':
    optimizer = None
  elif active_optimizer_name == 'adam':
    optimizer = make_adam_optimizer()
  elif active_optimizer_name == 'lamb':
    optimizer = make_lamb_optimizer()
  elif active_optimizer_name == 'kfac':
    optimizer = make_kfac_optimizer()
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  elif scheduled_adam_kfac:
    adam_optimizer = make_adam_optimizer()
    kfac_optimizer = make_kfac_optimizer()
    if t_init >= switch_iteration:
      optimizer = kfac_optimizer
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      opt_state = optimizer.init(params, subkeys, data)
      current_optimizer_name = 'kfac'
    else:
      optimizer = adam_optimizer
      opt_state = jax.pmap(optimizer.init)(params)
      opt_state = opt_state_ckpt or opt_state
      current_optimizer_name = 'adam'
  else:
    raise ValueError(f'Not a recognized optimizer: {active_optimizer_name}')

  initial_stage = training_stages[0] if training_stages else None

  if scheduled_adam_kfac:
    step = make_step_for_stage(initial_stage)
    logging.info(
        'Using adam_kfac optimizer: Adam until step %d, then KFAC.',
        switch_iteration)
  elif not optimizer:
    opt_state = None
    step = make_step_for_stage(initial_stage)
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_step_for_stage(initial_stage)
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_step_for_stage(initial_stage)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')

  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))

  burn_in_iterations = (
      _stage_iterations(burn_stage) if burn_stage is not None
      else cfg.mcmc.burn_in)

  if t_init == 0 and burn_in_iterations > 0: # MCMC burn in
    burn_in_name = burn_stage.get('name', 'MCMC burn-in') if burn_stage else 'MCMC burn-in'
    logging.info('Burning in MCMC chain for %d steps', burn_in_iterations)

    burn_in_step = make_training_step(
        mcmc_step=make_stage_mcmc_step(burn_stage),
        optimizer_step=null_update)

    burn_in_progress = _progress_bar(
        range(burn_in_iterations), desc=burn_in_name)
    burn_in_stats_frequency = max(
        1, int(cfg.log.get('burn_in_stats_frequency', 10)))
    burn_in_pmove = None
    burn_in_esjd_per_particle = None
    burn_in_esjd_per_moved_particle = None
    for t in burn_in_progress:
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      (data, params, _, _, _, burn_in_pmove, burn_in_esjd_per_particle,
       burn_in_esjd_per_moved_particle) = burn_in_step(
          data,
          params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
      adapt_frequency = burn_stage.get(
          'adapt_frequency', cfg.mcmc.adapt_frequency) if burn_stage else cfg.mcmc.adapt_frequency
      if (_stage_adapts_width(cfg, burn_stage) and
          t % adapt_frequency == 0):
        mcmc_width = _adaptive_mcmc_width_update(
            cfg, mcmc_width, burn_in_pmove[0], burn_stage)
      if (hasattr(burn_in_progress, 'set_postfix') and
          t % burn_in_stats_frequency == 0):
        burn_in_progress.set_postfix(
            pmove=f'{float(np.asarray(burn_in_pmove[0])):.2f}',
            w=f'{float(np.asarray(mcmc_width[0])):.3g}',
            esjd_m=(
                f'{float(np.asarray(burn_in_esjd_per_moved_particle[0])):.3g}'))
    logging.info('Completed burn-in MCMC steps')
    if burn_in_pmove is not None:
      logging.info(
          ('Final burn-in MCMC stats: pmove=%0.3f, width=%g, '
           'esjd_per_particle=%g, esjd_per_moved_particle=%g'),
          float(np.asarray(burn_in_pmove[0])),
          float(np.asarray(mcmc_width[0])),
          float(np.asarray(burn_in_esjd_per_particle[0])),
          float(np.asarray(burn_in_esjd_per_moved_particle[0])))
  if t_init == 0:
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])
    initial_energy_value = float(np.asarray(initial_energy[0]))
    if not np.isfinite(initial_energy_value):
      raise FloatingPointError(
          f"Initial local energy is nonfinite: {initial_energy_value}. "
          "Aborting before optimizer updates can poison the checkpoint.")

  time_of_last_ckpt = time.time()
  weighted_stats = None

  if active_optimizer_name == 'none' and opt_state_ckpt is not None:
    # If opt_state_ckpt is None, then we're restarting from a previous inference
    # run (most likely due to preemption) and so should continue from the last
    # iteration in the checkpoint. Otherwise, starting an inference run from a
    # training run.
    logging.info('No optimizer provided. Assuming inference run.')
    logging.info('Setting initial iteration to 0.')
    t_init = 0

    if not ckpt_restore_filename:
      logging.info('Inference aborted because no checkpoint loaded.')
      raise Exception("Inference aborted because no checkpoint loaded.")

  if writer_manager is None:
    if active_optimizer_name == 'none':
      stats_name = "inference_stats"
    else:
      stats_name = "train_stats"
    # if ${stats_name}.csv already exists from previous run, rename it to train_stats_N.cnv
    writers.rename_file(stats_name, cfg.log.save_path, file_extension="csv")

    writer_manager = writers.Writer(
        name=stats_name,
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False)
    
  with writer_manager as writer:
    # Main training loop
    num_resets = 0  # used if reset_if_nan is true
    current_stage_index = None
    stage_start = 0
    current_stage = None
    current_stage_name = 'training'

    def stage_for_step(t):
      if not training_stages:
        return None, None, 0, 'training'
      start = 0
      for index, stage in enumerate(training_stages):
        end = start + _stage_iterations(stage)
        if t < end:
          return index, stage, start, stage.get('name', f'stage_{index}')
        start = end
      stage = training_stages[-1]
      index = len(training_stages) - 1
      return index, stage, start - _stage_iterations(stage), stage.get(
          'name', f'stage_{len(training_stages) - 1}')

    steps = range(t_init, cfg.optim.iterations)
    train_progress = _progress_bar(
        steps, desc='Training', total=cfg.optim.iterations - t_init)
    for t in train_progress:
      stage_index, current_stage, stage_start, current_stage_name = (
          stage_for_step(t))
      if stage_index != current_stage_index:
        current_stage_index = stage_index
        step = make_step_for_stage(current_stage)
        logging.info(
            'Starting training stage %s at step %d', current_stage_name, t)

      if (scheduled_adam_kfac and current_optimizer_name == 'adam' and
          t >= switch_iteration):
        logging.info('Switching optimizer from Adam to KFAC at step %d.', t)
        optimizer = kfac_optimizer
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        opt_state = optimizer.init(params, subkeys, data)
        step = make_step_for_stage(current_stage)
        current_optimizer_name = 'kfac'

      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      (data, params, opt_state, loss, aux_data, pmove, esjd_per_particle,
       esjd_per_moved_particle) = step(
          data,
          params,
          opt_state,
          subkeys,
          mcmc_width)
      
      # due to pmean, loss, and pmove should be the same across devices.
      loss = loss[0]
      has_loss = aux_data is not None
      if has_loss:
        # per batch variance isn't informative. Use weighted mean and variance instead.
        weighted_stats = statistics.exponentialy_weighted_stats(
            alpha=0.1, observation=loss, previous_stats=weighted_stats)

      pmove = pmove[0]
      esjd_per_particle = esjd_per_particle[0]
      esjd_per_moved_particle = esjd_per_moved_particle[0]

      # variance = aux_data.variance[0] 
      local_std = jnp.sqrt(aux_data.variance[0]) if has_loss else jnp.nan

      # Update MCMC move width
      local_stage_step = t - stage_start
      if _stage_adapts_width(cfg, current_stage):
        stage_adaptive_steps = (
            current_stage.get('adaptive_steps', _stage_iterations(
                current_stage))
            if current_stage is not None
            else cfg.mcmc.get('adaptive_steps', cfg.optim.iterations))
        stage_adapt_frequency = (
            current_stage.get('adapt_frequency', cfg.mcmc.adapt_frequency)
            if current_stage is not None
            else cfg.mcmc.adapt_frequency)
        pred = (
            (local_stage_step < stage_adaptive_steps) &
            (jnp.remainder(local_stage_step, stage_adapt_frequency) == 0)
        )
        mcmc_width = jax.lax.cond(
            pred,
            lambda w: _adaptive_mcmc_width_update(
                cfg, w, pmove, current_stage),
            lambda w: w,
            mcmc_width)

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if active_optimizer_name != 'none':
          tree['optim'] = opt_state
        try:
          chex.assert_tree_all_finite(tree)
          num_resets = 0  # Reset counter if check passes
        except AssertionError as e:
          if cfg.optim.reset_if_nan:  # Allow a certain number of NaNs
            num_resets += 1
            if num_resets > 100:
              raise e
          else:
            raise e

      # Logging
      if t % cfg.log.stats_frequency == 0:
        num_particles = sum(nspins)
        checkpoint_score = (
            _best_checkpoint_score(
                cfg.log.get('best_checkpoint_metric', 'ewmean'),
                loss,
                local_std,
                weighted_stats,
                num_particles=num_particles,
                std_weight=cfg.log.get('best_checkpoint_std_weight', 1.0))
            if has_loss else np.nan)
        # write to train_stats
        writer_kwargs = {
            'step': t,
            'stage': current_stage_name,
            'energy': np.asarray(loss) if has_loss else np.nan,
            'ewmean': np.asarray(weighted_stats.mean) if has_loss else np.nan,
            'ewvar': np.asarray(
                weighted_stats.variance) if has_loss else np.nan,
            'pmove': np.asarray(pmove),
            'locstd': np.asarray(local_std),
            'mcmc_width': np.asarray(mcmc_width[0]),
            'mcmc_esjd_per_particle': np.asarray(esjd_per_particle),
            'mcmc_esjd_per_moved_particle': np.asarray(
                esjd_per_moved_particle),
        }
        writer.write(t, **writer_kwargs)

        # log training data
        stage_total_steps = (
            _stage_iterations(current_stage) if current_stage is not None
            else cfg.optim.iterations)
        local_stage_step = t - stage_start
        width = float(np.asarray(mcmc_width[0]))
        pmove_value = float(np.asarray(pmove))
        esjd_per_particle_value = float(np.asarray(esjd_per_particle))
        esjd_per_moved_value = float(np.asarray(esjd_per_moved_particle))
        if has_loss:
          energy_per_particle = float(np.asarray(loss / num_particles))
          ewmean_per_particle = float(
              np.asarray(weighted_stats.mean / num_particles))
          std_per_particle = float(np.asarray(local_std / num_particles))
          ewstd_per_particle = float(
              np.asarray(jnp.sqrt(weighted_stats.variance) / num_particles))
          logging.info(
              ('[stage=%s step=%05d local=%05d/%05d] '
               'E/N=%0.6g ewE/N=%0.6g std/N=%0.6g ewstd/N=%0.6g '
               'pmove=%0.3f width=%0.6g esjd/N=%0.6g esjd/moved=%0.6g '
               'score=%0.6g'),
              current_stage_name,
              t,
              local_stage_step,
              stage_total_steps,
              energy_per_particle,
              ewmean_per_particle,
              std_per_particle,
              ewstd_per_particle,
              pmove_value,
              width,
              esjd_per_particle_value,
              esjd_per_moved_value,
              checkpoint_score)
        else:
          logging.info(
              ('[stage=%s step=%05d local=%05d/%05d] sampler '
               'pmove=%0.3f width=%0.6g esjd/N=%0.6g esjd/moved=%0.6g'),
              current_stage_name,
              t,
              local_stage_step,
              stage_total_steps,
              pmove_value,
              width,
              esjd_per_particle_value,
              esjd_per_moved_value)
        if hasattr(train_progress, 'set_postfix'):
          postfix = {
              'stage': current_stage_name,
              'pmove': f'{float(np.asarray(pmove)):.2f}',
              'w': f'{float(np.asarray(mcmc_width[0])):.3g}',
              'esjd_m': f'{float(np.asarray(esjd_per_moved_particle)):.3g}',
          }
          if has_loss:
            postfix |= {
                'E_N': f'{float(np.asarray(loss / num_particles)):.4g}',
                'std_N': f'{float(np.asarray(local_std / num_particles)):.4g}',
                'score': f'{checkpoint_score:.4g}',
            }
          train_progress.set_postfix(**postfix)
        
      # Checkpointing every cfg.log.save_frequency minutes and at final iteration
      if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60 or t == (cfg.optim.iterations - 1):
        ckpt_filename = checkpoint.save(
            ckpt_save_path, t, data, params, opt_state, mcmc_width)
        checkpoint.prune_checkpoints(
            ckpt_save_path,
            keep_latest=cfg.log.get('keep_latest_checkpoints', 5))
        if has_loss and t >= cfg.log.get('best_checkpoint_min_step', 500):
          score = _best_checkpoint_score(
              cfg.log.get('best_checkpoint_metric', 'ewmean'),
              loss,
              local_std,
              weighted_stats,
              num_particles=sum(nspins),
              std_weight=cfg.log.get('best_checkpoint_std_weight', 1.0))
          checkpoint.update_best_checkpoints(
              ckpt_filename,
              ckpt_save_path,
              step=t,
              score=score,
              keep_best=cfg.log.get('keep_best_checkpoints', 3))
        time_of_last_ckpt = time.time()
