"""
Seed Management Utility for Reproducible RL Experiments
========================================================
Ensures deterministic behavior across Python, NumPy, PyTorch, and CUDA.

Usage:
    from utils.seed import set_global_seed, get_seed_from_args

    # Set seed everywhere
    set_global_seed(42)

    # Or parse from CLI
    seed = get_seed_from_args()
    set_global_seed(seed)

Reference:
    Henderson et al., "Deep Reinforcement Learning that Matters" (2018)
    https://arxiv.org/abs/1709.06560
"""

import os
import random
import argparse


def set_global_seed(seed: int = 42, deterministic_cudnn: bool = True) -> dict:
    """
    Set random seed for reproducibility across all frameworks.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, set CuDNN to deterministic mode.
            This may reduce performance but ensures reproducibility.

    Returns:
        Dictionary with environment info for logging.
    """
    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    torch_info = {}
    try:
        import torch

        torch.manual_seed(seed)
        torch_info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch_info["cuda_available"] = True
            torch_info["cuda_version"] = torch.version.cuda
            torch_info["gpu_count"] = torch.cuda.device_count()
            torch_info["gpu_name"] = (
                torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
            )

            if deterministic_cudnn:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
        else:
            torch_info["cuda_available"] = False
    except ImportError:
        pass

    # TensorFlow (if used)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    env_info = {
        "seed": seed,
        "python_hash_seed": str(seed),
        "deterministic_cudnn": deterministic_cudnn,
        **torch_info,
    }

    return env_info


def get_seed_from_args(default: int = 42) -> int:
    """Parse --seed from command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=default)
    args, _ = parser.parse_known_args()
    return args.seed


def get_environment_info() -> dict:
    """
    Collect environment information for reproducibility logging.
    Logs GPU type, CUDA version, library versions, etc.
    """
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }

    # PyTorch
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem
                info[f"gpu_{i}_memory_gb"] = round(mem / (1024**3), 1)
    except ImportError:
        info["torch_version"] = "not installed"

    # Transformers
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    # TRL
    try:
        import trl

        info["trl_version"] = trl.__version__
    except ImportError:
        pass

    # NumPy
    try:
        import numpy as np

        info["numpy_version"] = np.__version__
    except ImportError:
        pass

    # Datasets
    try:
        import datasets

        info["datasets_version"] = datasets.__version__
    except ImportError:
        pass

    return info


def log_experiment_metadata(
    experiment_name: str,
    seed: int,
    hyperparameters: dict,
    output_dir: str = "./results",
) -> str:
    """
    Log full experiment metadata to a JSON file for reproducibility.

    Args:
        experiment_name: Name of the experiment
        seed: Random seed used
        hyperparameters: Dict of all hyperparameters
        output_dir: Directory to save metadata

    Returns:
        Path to the saved metadata file.
    """
    import json
    import datetime

    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "hyperparameters": hyperparameters,
        "environment": get_environment_info(),
    }

    filepath = os.path.join(output_dir, f"{experiment_name}_seed{seed}_metadata.json")

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Experiment metadata saved to {filepath}")
    return filepath
