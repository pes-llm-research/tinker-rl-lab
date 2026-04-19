"""
OpenRLHF Integration for tinker-rl-lab

OpenRLHF: An Easy-to-use, Scalable and High-performance Agentic RL Framework
GitHub: https://github.com/OpenRLHF/OpenRLHF

Features:
- Ray + vLLM distributed architecture
- PPO, DAPO, REINFORCE++ algorithms
- Async RL training
- Multi-GPU and multi-node support
"""

__version__ = "0.9.0"

from .trainer import OpenRLHFTrainer, run_openrlhf_training, run
from .config import OpenRLHFConfig

__all__ = ["OpenRLHFTrainer", "OpenRLHFConfig", "run_openrlhf_training", "run"]
