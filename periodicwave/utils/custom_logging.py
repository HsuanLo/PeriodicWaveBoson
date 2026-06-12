# Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
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

"""Logging functionalities."""

import logging
import os
import platform
import subprocess
from sys import argv

import jax
import jax.numpy as jnp
import jaxlib
import kfac_jax
import numpy as np
import json


def _read_env(name):
    return os.environ.get(name, "<unset>")


def _log_section(title):
    logging.info("")
    logging.info("=== %s ===", title)


def _run_command(command):
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except FileNotFoundError:
        return f"Command not found: {command[0]}"
    except subprocess.CalledProcessError as exc:
        return (
            f"Command failed ({exc.returncode}): {' '.join(command)}\n"
            f"{exc.output.strip()}")

def log_device_info(log_file="device_info.log"):
    """
    Logs important device and environment information when training a neural network with JAX.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create if a directory path is provided
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format="%(asctime)s - %(message)s")
    
    _log_section("Runtime Environment")
    logging.info("Command line: %s", " ".join(argv))
    logging.info("Working directory: %s", os.getcwd())
    logging.info("Host: %s", platform.node())
    logging.info("Python version: %s", platform.python_version())
    logging.info("System: %s %s", platform.system(), platform.version())
    logging.info("Processor: %s", platform.processor())
    logging.info("Numpy version: %s", np.__version__)

    _log_section("Scheduler Environment")
    for name in (
        "LLSUB_RANK",
        "LLSUB_SIZE",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
    ):
        logging.info("%s: %s", name, _read_env(name))

    _log_section("JAX Device Visibility")
    jax_version = jax.__version__
    jaxlib_version = jaxlib.__version__
    kfac_jax_version = kfac_jax.__version__
    available_devices = jax.devices()
    local_devices = jax.local_devices()
    platform_name = jax.default_backend()
    precision = jnp.finfo(jnp.float32).dtype.name  # Default precision
    matmul_precision = jax.config.jax_default_matmul_precision
    jax_enable_x64 = jax.config.read("jax_enable_x64")  # Check if 64-bit precision is enabled
    logging.info("JAX version: %s", jax_version)
    logging.info("JAXlib version: %s", jaxlib_version)
    logging.info("KFAC-JAX version: %s", kfac_jax_version)
    logging.info("Platform: %s", platform_name)
    logging.info("jax_enable_x64: %s", jax_enable_x64)
    logging.info("Default precision: %s", precision)
    logging.info("Default matmul precision: %s", matmul_precision)
    logging.info("CUDA_VISIBLE_DEVICES: %s", _read_env("CUDA_VISIBLE_DEVICES"))
    logging.info(
        "XLA_PYTHON_CLIENT_PREALLOCATE: %s",
        _read_env("XLA_PYTHON_CLIENT_PREALLOCATE"))
    logging.info(
        "XLA_PYTHON_CLIENT_MEM_FRACTION: %s",
        _read_env("XLA_PYTHON_CLIENT_MEM_FRACTION"))
    logging.info(
        "JAX process index/count: %d/%d",
        jax.process_index(),
        jax.process_count())
    logging.info("JAX local device count: %d", jax.local_device_count())
    logging.info("JAX global device count: %d", len(available_devices))
    logging.info("JAX local devices: %s", [str(device) for device in local_devices])
    logging.info("JAX global devices: %s", [str(device) for device in available_devices])
    logging.info(
        "JAX visible GPU kinds: %s",
        [device.device_kind for device in available_devices]
        if "gpu" in platform_name else "No GPU backend detected.")

    _log_section("Node NVIDIA State")
    logging.info(
        "nvidia-smi -L output (node-level, not necessarily JAX-visible):\n%s",
        _run_command(["nvidia-smi", "-L"]))
    logging.info(
        "nvidia-smi query (node-level, not necessarily JAX-visible):\n%s",
        _run_command([
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ]))

    _log_section("Training Log")
    
    print(f"Device and environment info logged to {log_file}")

def save_config_dict_as_json(config_dict, file_path):
    """
    Saves a ConfigDict object to a JSON file using its `to_json` method.
    
    Args:
        config_dict (ConfigDict): The ConfigDict object to save.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, 'w') as json_file:
        json_file.write(config_dict.to_json_best_effort(indent=4))
    print(f"Configuration saved as JSON to {file_path}")

def load_config_dict_from_json(file_path, config_class):
    """
    Loads a ConfigDict object from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file.
        config_class (type): The class of the ConfigDict (e.g., `ConfigDict`).
        
    Returns:
        ConfigDict: The loaded ConfigDict object.
    """
    with open(file_path, 'r') as json_file:
        json_content = json_file.read()
    config_dict = config_class.from_json(json_content)
    print(f"Configuration loaded from {file_path}")
    return config_dict

# Example usage
if __name__ == "__main__":
    program_name = argv[0]
    logfile_name = argv[1]
    log_device_info(logfile_name + '.log')
