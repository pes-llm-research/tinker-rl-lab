"""
verl Integration for tinker-rl-lab

verl: Volcano Engine Reinforcement Learning for LLMs
GitHub: https://github.com/verl-project/verl

A flexible, efficient and production-ready RL training framework designed for
large language models (LLMs) post-training.

Features:
- HybridFlow programming model for RL
- Multi-GPU and multi-node support via Ray
- PPO, GRPO, and other RL algorithms
- Integration with vLLM for high-throughput inference
"""

__version__ = "0.5.0"

from .trainer import VERLTrainer, run_verl_training, run
from .config import VERLConfig

__all__ = ["VERLTrainer", "VERLConfig", "run_verl_training", "run"]
