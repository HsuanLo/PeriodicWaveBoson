# Copyright 2022 DeepMind Technologies Limited.
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
# - Streamlined for materials calculations

""" Default base configuration for VMC calculations for periodic systems. """

import ml_collections
from ml_collections import config_dict

def default() -> ml_collections.ConfigDict:
  """Create set of default parameters for running qmc.py.

  Note: placeholder `cfg.system.bosons` must be replaced with appropriate values.

  Returns:
    ml_collections.ConfigDict containing default settings.
  """
  # wavefunction output.
  cfg = ml_collections.ConfigDict({
      'batch_size': 1024,  # Default value that empirically works well for two-dimensional problems.
      'optim': {
          # Objective type. Only 'vmc' implemented.
          # 'vmc': minimise <H> by standard VMC energy minimization
          'objective': 'vmc',
          'iterations': 1000000,  # number of iterations
          'optimizer': 'kfac',  # one of adam, adam_kfac, kfac, lamb, none
          'laplacian': 'default',  # one of default or folx (for forward lapl)
          'lr': {
              'rate':  0.05,  # learning rate
              'decay': 1.0,  # exponent of learning rate decay
              'delay': 10000.0,  # term that sets the scale of the rate decay
          },
          # If greater than zero, scale (at which to clip local energy) in units
          # of the mean deviation from the mean.
          'clip_local_energy': 5.0,
          # If true, center the clipping window around the median rather than
          # the mean. More "correct" for removing outliers, but also potentially
          # slow, especially with multihost training.
          'clip_median': False,
          # If true, center the local energy differences in the gradient at the
          # average clipped energy rather than average energy, guaranteeing that
          # the average energy difference will be zero in each batch.
          'center_at_clip': True,
          # If true, keep the parameters and optimizer state from the previous
          # step and revert them if they become NaN after an update. 
          'reset_if_nan': False,
          # KFAC hyperparameters. See KFAC documentation for details.
          'kfac': {
              'invert_every': 1,
              'cov_update_every': 1,
              'damping': 0.001,
              'cov_ema_decay': 0.95,
              'momentum': 0.0,
              'momentum_type': 'regular',
              # Warning: adaptive damping is not currently available.
              'min_damping': 1.0e-6,
              'norm_constraint': 0.001,
              'mean_center': True,
              'l2_reg': 0.0,
              'register_only_generic': False,
          },
          # ADAM hyperparameters. See optax documentation for details.
          'adam': {
              'b1': 0.9,
              'b2': 0.999,
              'eps': 1.0e-8,
              'eps_root': 0.0,
          },
          # Two-stage optimizer: Adam first, then initialize KFAC from the
          # current parameters and walkers at switch_iteration.
          'adam_kfac': {
              'switch_iteration': 1000,
              'kfac_lr_rate': 0.01,
              'kfac_lr_delay': 10000.0,
              'kfac_lr_decay': 1.0,
          },
      },
      'log': {
          'stats_frequency': 1,  # iterations between logging of stats
          'save_frequency': 10.0,  # minutes between saving network params
          # Path to save/restore network to/from. If falsy,
          # creates a timestamped directory in the working directory.
          'save_path': '',
          # Path containing checkpoint to restore network from.
          # Ignored if falsy or save_path contains a checkpoint.
          'restore_path': '',
      },
      'system': {
          # Specify the system by setting variables below.
          # Bilayer boson occupations, e.g. (top_layer, bottom_layer).
          'bosons': tuple(),
          # Dimensionality. 
          'ndim': 2,
          # String set to module.make_local_energy, where make_local_energy is a
          # callable (type: MakeLocalEnergy) which creates a function which
          # evaluates the local energy and module is the absolute module
          # containing make_local_energy.
          # If not set, hamiltonian.local_energy is used (only kinetic energy).
          'make_local_energy_fn': '',
          # Additional kwargs to pass into make_local_energy_fn.
          'make_local_energy_kwargs': {},
          # If periodic boundary conditions are used, store lattice:
          'pbc_lattice': None,
      },
      'mcmc': {
          # Note: HMC options are not currently used.
          # Number of burn in steps after pretraining.  If zero do not burn in
          # or reinitialize walkers.
          'burn_in': 300,
          'steps': 30,  # Number of MCMC steps to make between network updates.
          # Width of Gaussian used to generate initial boson configurations.
          'init_width': 1.0,
          # Width of Gaussian used for random moves for RMW or step size for HMC.
          'move_width': 0.1,
          # How to update move_width during runtime.
          # const: move_width remains constant
          # adaptive: increases or reduces move_width depending on pmove (default)
          'move_width_updater': 'adaptive',
          # Number of steps after which to update the adaptive MCMC step size
          'adapt_frequency': 100,
          # Target acceptance for adaptive random-walk Metropolis moves.
          'target_acceptance': 0.5,
          # Log-space adaptation rate for move_width.
          'adapt_rate': 0.02,
          # Bounds for adaptive move_width.
          'min_move_width': 1.0e-4,
          'max_move_width': 1.0,
          # Emergency adaptation thresholds/factors for very poor acceptance.
          'low_acceptance': 0.05,
          'moderate_low_acceptance': 0.20,
          'high_acceptance': 0.90,
          'low_acceptance_factor': 0.5,
          'moderate_low_acceptance_factor': 0.8,
          'high_acceptance_factor': 1.02,
          # Stop adapting the MCMC move width after this many optimization
          # iterations. This freezes the proposal scale for cleaner statistics.
          'adaptive_steps': 1000,
          'blocks': 1,  # Number of blocks to split the MCMC sampling into
      },
      'network': {
          'network_type': 'BosonNet',
          # If true, the network outputs complex numbers rather than real.
          'complex': False,
          # Symmetric bosonic network, not a determinant wavefunction.
          'BosonNet': {
              'architecture': "DeepSets", # one of "DeepSets", "Attention", or "Transformer"
              'num_layers': 3,
              'mlp_dim': 64,
              'num_heads': 4,
              'attn_dim': 16,
              'value_dim': 16,
              'num_perceptrons_per_layer': 2,
              'use_layer_norm': True,
              'mlp_activation_fct': "GELU",
          },
      },
      'debug': {
          # Check optimizer state, parameters and loss and raise an exception if
          # NaN is found.
          'check_nan': False,
          'deterministic': False,  # Use a deterministic seed.
      },
  })

  return cfg
