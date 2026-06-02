# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0.

"""Run a bilayer boson NN-VMC calculation with xy PBC."""

import os
from time import time

from absl import logging
import jax
import numpy as np

from periodicwave import base_config
from periodicwave import train
from periodicwave.pbc import lattices
from periodicwave.utils import custom_logging
from periodicwave.utils import writers


print("Jax Devices:", jax.devices())
jax.config.update("jax_default_matmul_precision", "float32")


def _env_float(name, default):
  value = os.environ.get(name)
  if value is None:
    return default
  return float(value)


# --------------------------- Physical parameters ---------------------------
num_bosons = 24
layer_occupations = (12, 12)
layer_separation = _env_float("SCAN_D", 1.0)
dipole_strength = 20.0
supercell_shape = "sq"
density_rs = _env_float("SCAN_RS", 0.5)

if sum(layer_occupations) != num_bosons:
  raise ValueError("layer_occupations must sum to num_bosons.")

layer_assignment = np.array(
    [1.0] * layer_occupations[0] + [-1.0] * layer_occupations[1])

if supercell_shape == "tri":
  supercell_a = density_rs * np.sqrt(2 * np.pi / np.sqrt(3) * num_bosons)
  supercell_lattice, _ = lattices._triangular_lattice_vecs_periodic_potential(
      supercell_a, 1)
elif supercell_shape == "sq":
  supercell_a = density_rs * np.sqrt(np.pi * num_bosons)
  supercell_lattice = lattices._square_lattice_vecs(supercell_a)
else:
  raise NotImplementedError(f"Unknown supercell_shape: {supercell_shape}")

cell_area = abs(np.linalg.det(supercell_lattice))


# --------------------------- Set up config file ---------------------------
cfg = base_config.default()
cfg.batch_size = 2048

cfg.system.bosons = (num_bosons, 0)
cfg.system.ndim = 2
cfg.system.pbc_lattice = supercell_lattice
cfg.system.make_local_energy_fn = "periodicwave.pbc.bilayer_hamiltonian.local_energy"
cfg.system.make_local_energy_kwargs = {
    "lattice": supercell_lattice,
    "layer_separation": layer_separation,
    "potential_type": "Dipolar",
    "potential_kwargs": {
        "dipole_strength": dipole_strength,
        "softening": 1e-2,
        "use_ewald": True,
        "ewald_alpha": 10.0 / np.sqrt(cell_area),
        "ewald_real_cut": 2,
        "ewald_kmax": 12,
        "ewald_geometry": "xy_periodic_open_z",
    },
    "kinetic_kwargs": {"laplacian_method": "folx"},
}

cfg.network.network_type = "BosonNet"
cfg.network.complex = False
cfg.network.BosonNet.architecture = "Transformer"
cfg.network.BosonNet.num_layers = 3
cfg.network.BosonNet.mlp_dim = 64
cfg.network.BosonNet.num_heads = 4
cfg.network.BosonNet.attn_dim = 16
cfg.network.BosonNet.value_dim = 16
cfg.network.BosonNet.num_perceptrons_per_layer = 2
cfg.network.BosonNet.use_layer_norm = True
cfg.network.BosonNet.mlp_activation_fct = "GELU"

cfg.mcmc.burn_in = 2000
cfg.mcmc.steps = 20
cfg.mcmc.init_width = 2.0
cfg.mcmc.move_width = 0.05
cfg.mcmc.move_width_updater = "adaptive"
cfg.mcmc.adapt_frequency = 10
cfg.mcmc.target_acceptance = 0.5
cfg.mcmc.adapt_rate = 0.05
cfg.mcmc.min_move_width = 1e-3
cfg.mcmc.max_move_width = 2.0
cfg.mcmc.adaptive_steps = 1500

cfg.optim.optimizer = "adam_kfac"
cfg.optim.iterations = 5000
cfg.optim.lr.rate = 1e-4
cfg.optim.lr.delay = 1000
cfg.optim.lr.decay = 1.0
cfg.optim.adam_kfac.switch_iteration = 500
cfg.optim.adam_kfac.kfac_lr_rate = 0.01
cfg.optim.adam_kfac.kfac_lr_delay = 1000.0
cfg.optim.adam_kfac.kfac_lr_decay = 1.0

cfg.log.save_frequency = 10.0
cfg.debug.deterministic = True

folder_name = (
    "results/bilayer-bosons/"
    f"BosonNet/N{num_bosons}_layers{layer_occupations[0]}_{layer_occupations[1]}"
    f"_rs{density_rs}_d{layer_separation}_D{dipole_strength}_{supercell_shape}"
)
cfg.log.save_path = folder_name

writers.rename_file("device_info", folder_name, file_extension="log")
custom_logging.log_device_info(folder_name + "/device_info.log")
writers.rename_file("config", folder_name, file_extension="json")
custom_logging.save_config_dict_as_json(cfg, cfg.log.save_path + "/config.json")

t_init = time()
train.train(cfg, layer_assignment=layer_assignment)
logging.info("Training completed after t [s] = " + str(int(time() - t_init)))
