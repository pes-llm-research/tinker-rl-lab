# Tinker RL Lab

A consolidated research repository for Reinforcement Learning experiments with Large Language Models, integrating multiple RL frameworks and compute backends.

## Overview

This repository consolidates multiple research projects focused on:
- **Multiple RL Frameworks**: Tinker/SkyRL, verl, OpenRLHF, TRL (HuggingFace)
- **Multiple Compute Backends**: Local GPU, vast.ai, Google Colab
- **Multiple Environments**: Atropos, GSM8K, Math, HumanEval, Tool Use

## Repository Structure

```
tinker-rl-lab/
├── skyrl/               # SkyRL tx integration (Local Tinker API)
│   ├── backends/         # vast.ai and Colab runners
│   ├── configs/          # YAML configurations
│   └── notebooks/         # Colab notebooks
│
├── atropos/              # Tinker-Atropos integration
│   ├── tinker_atropos/   # Core package
│   │   ├── environments/ # GSM8K, Math, LogP steering
│   │   ├── trainer.py    # Tinker trainer
│   │   └── config.py     # Configuration management
│   ├── configs/          # YAML configurations
│   └── notebooks/        # Analysis notebooks & GRPO results
│
├── verl/                 # Volcano Engine RL integration
│   └── trainer.py
│
├── openrlhf/             # OpenRLHF integration
│   └── trainer.py
│
├── trl_integrations/     # HuggingFace TRL integration
│   └── trainer.py
│
├── unified/              # Unified launcher for all frameworks
│   └── launcher.py
│
├── experiments/           # Tinker RL Cookbook experiments
│   ├── notebooks/        # Jupyter notebooks for each experiment
│   ├── implementations/  # RL implementations (PPO, DPO, GRPO, etc.)
│   ├── results/          # Training metrics
│   └── tinker-runs/     # Training logs and scripts
│
├── agentic-rl-finetuning/ # Agentic RL fine-tuning research
├── capstone-literature-survey/ # Literature Survey: RL for LLMs (GRPO Scaling)
└── reports/              # Final capstone report and paper
```

## Components

### 1. Experiments (Tinker RL Cookbook)

PES LLM Research Project experiments using the [Tinker](https://thinkingmachines.ai/tinker) platform.

| Recipe | Description | Status |
|--------|-------------|--------|
| Math RL (Arithmetic) | Train model to add numbers | Complete - 100% accuracy |
| Chat SL | Supervised fine-tuning on NoRobots | Complete |
| Preference Shorter | Train for concise responses | Complete |
| Distillation Off-Policy | SFT on OpenThoughts3 | Complete |
| Distillation On-Policy | KL minimization to teacher | Complete |
| Math RL (GSM8K) | Word problem solving | Complete |

**Key Results:**
- Arithmetic: 69.5% → 100% accuracy in ~20 steps
- Preference learning effectively shapes response style
- Distillation transfers knowledge efficiently

### 2. Atropos Integration

Integration layer connecting [Atropos](https://github.com/NousResearch/atropos) with the Tinker API.

Features:
- Use any Atropos environment with Tinker training
- Built-in GSM8K and Math environments
- LoRA-based fine-tuning with configurable parameters
- Checkpoint management and weight downloading

### 3. SkyRL (Local Tinker API)

SkyRL tx implements the Tinker API locally on your own GPUs. Any tinker-cookbook recipe works without cloud API.

Features:
- Local Tinker API server implementation
- GRPO, PPO, REINFORCE algorithms
- vLLM/SGLang inference
- vast.ai and Colab backends

### 4. verl (Volcano Engine RL)

Production-ready RL training framework with HybridFlow programming model.

Features:
- Multi-GPU and multi-node via Ray
- PPO, GRPO, REINFORCE
- High-throughput vLLM inference

### 5. OpenRLHF

Scalable agentic RL framework with Ray + vLLM distributed architecture.

Features:
- PPO, DAPO, REINFORCE++
- Async RL training
- Multi-GPU and multi-node support

### 6. TRL (HuggingFace)

HuggingFace's full-stack RL library with easy model integration.

Features:
- GRPO, PPO, DPO, Reward Modeling
- Works with any HF model
- Single GPU to multi-GPU (DeepSpeed)

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python3 -m venv tinker-env
source tinker-env/bin/activate

# Install dependencies
pip install tinker tinker-cookbook atropos
```

### Running Tinker Experiments

```bash
export TINKER_API_KEY="your-key-here"

# Math RL
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="meta-llama/Llama-3.2-1B" \
    env=arithmetic

# Chat SFT
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name="meta-llama/Llama-3.2-1B"
```

### Running Atropos + Tinker

```bash
# Terminal 1: Start Atropos API
run-api

# Terminal 2: Start training
export TINKER_API_KEY="your-key"
python atropos/launch_training.py --config atropos/configs/default.yaml

# Terminal 3: Start environment
python atropos/tinker_atropos/environments/gsm8k_tinker.py serve \
    --config atropos/configs/default.yaml
```

### Running SkyRL (Local Tinker API)

```bash
# Start local Tinker API server
cd SkyRL/skyrl-train
uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model Qwen/Qwen2.5-1.5B-Instruct --port 8000

# In another terminal - run any tinker-cookbook recipe
export TINKER_API_KEY="tml-dummy"
export TINKER_BASE_URL="http://localhost:8000"
python -m tinker_cookbook.recipes.math_rl.train base_url=$TINKER_BASE_URL ...
```

### Running on vast.ai

```bash
# SkyRL on vast.ai
cd skyrl/backends
./vastai_launch.sh --model Qwen/Qwen2.5-1.5B-Instruct

# Or use Python launcher
python -m skyrl.backends.vastai_runner --model Qwen/Qwen2.5-1.5B-Instruct
```

### Running with Unified Launcher

```bash
# Use any framework with unified launcher
python -m unified.launcher --framework skyrl --model Qwen/Qwen2.5-1.5B-Instruct
python -m unified.launcher --framework trl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm grpo
python -m unified.launcher --framework verl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm ppo
python -m unified.launcher --framework openrlhf --model Qwen/Qwen2.5-1.5B-Instruct
```

### Running in Google Colab

Open `skyrl/notebooks/skyrl_colab_training.ipynb` and run cells sequentially.

## Source Repositories

This repository consolidates all PES LLM Research projects:

| Original Repo | Description | Created |
|--------------|-------------|---------|
| [tinker-experiments](https://github.com/arvindcr4/tinker-experiments) | Tinker RL Cookbook experiments | Jan 2026 |
| [tinker-atropos](https://github.com/arvindcr4/tinker-atropos) | Atropos + Tinker integration | Mar 2026 |
| [rl](https://github.com/arvindcr4/rl) | RL Gym tasks and documentation | Aug 2025 |
| [rl_master](https://github.com/arvindcr4/rl_master) | Task execution and MCP tools | Nov 2025 |
| [agentic-rl-finetuning](https://github.com/pes-llm-research/agentic-rl-finetuning) | Agentic RL fine-tuning | Mar 2026 |
| [capstone-literature-survey](https://github.com/arvindcr4/capstone-literature-survey) | GRPO Scaling Literature Survey | Mar 2026 |

## Documentation

- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai)
- [Atropos GitHub](https://github.com/NousResearch/atropos)
- [Tinker Cookbook](https://github.com/thinkingmachines/tinker-cookbook)

## References

- [DeepCoder Blog Post](https://thinkingmachines.ai/blog/deepcoder)
- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation)

## Authors

**PES LLM Research Team**

- Arvind C R (PES University) &mdash; equal contribution
- Sandhya Jeyaraj (PES University) &mdash; equal contribution
- Madhu Kumara L (PES University)
- Mohammad Rafi (PES University)
- Dhruva N Murthy (PES University)
- Arumugam K (PES University)
- Anwesh Reddy Paduri (Great Learning / PES University) &mdash; project guide
- Narayana Darapaneni (Northwestern University / Great Learning) &mdash; project guide

Corresponding author: Arvind C R &lt;arvindcr4@gmail.com&gt;. Equal contribution denotes equal
technical and writing contribution; author order among the student team is alphabetical by given name after the two equal-contribution leads.
See [`CITATION.cff`](CITATION.cff) for the canonical BibTeX record.

## License

See individual component directories for license information.
