"""Modal GRPO Campaign — verl launcher.

Runs the Task 4 framework-gap config on a single Modal H100 via pypi verl:

    Qwen3-8B, seed=42, G=8, lr=1e-5, GSM8K[:500], 30 steps.

Sister script to ``experiments/modal/modal_grpo_trl.py``. Writes result.json
into ``/home/user/workspace/elevation_outputs/modal_verl_grpo.json`` for
pickup by ``experiments/results/aggregate_framework_comparison.py``.

Note: the repo's top-level ``verl/`` directory is a launcher wrapper package
that shares a name with pypi ``verl``. To avoid the shadow we clone the repo
into ``/root/repo`` (not on sys.path) and invoke pypi verl's Hydra CLI
directly from this file.
"""

import json
import os

import modal

app = modal.App("tinker-rl-verl-grpo")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("packaging", "wheel", "ninja", "setuptools")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "pyarrow",
        "pydantic>=2",
        "pyyaml",
    )
    .pip_install("flash-attn==2.7.0.post2", extra_options="--no-build-isolation")
    .pip_install(
        "vllm==0.7.2",
        "ray[default]>=2.34",
        "verl==0.3.0.post1",
        "transformers>=4.45",
        "datasets",
        "accelerate",
        "wandb",
        "huggingface_hub",
    )
    .env({
        "HF_TOKEN": HF_TOKEN,
        "WANDB_API_KEY": WANDB_KEY,
        "WANDB_PROJECT": "tinker-rl-lab-world-class",
    })
)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    secrets=[
        modal.Secret.from_dict({
            "HF_TOKEN": HF_TOKEN,
            "WANDB_API_KEY": WANDB_KEY,
            "WANDB_PROJECT": "tinker-rl-lab-world-class",
        })
    ],
)
def run_verl_qwen3_8b():
    """Execute the verl GRPO GSM8K-500 run on a single H100."""
    import re
    import subprocess
    import sys
    import time

    from datasets import load_dataset
    import numpy as np
    import wandb

    MODEL = "Qwen/Qwen3-8B"
    PROJECT = "tinker-rl-lab-world-class"
    RUN_NAME = "modal-verl-qwen3-8b-grpo-gsm8k500"
    WORK = "/root/verl-run"
    os.makedirs(WORK, exist_ok=True)

    # ---- GSM8K[:500] -> verl-compatible parquet (prompt + reward_model) ----
    SYS = "You are a math assistant. Solve step by step, then give your final answer inside \\boxed{}."
    ds = load_dataset("openai/gsm8k", "main", split="train[:500]")
    def to_verl(row):
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        gt = m.group(1).replace(",", "").strip() if m else ""
        return {
            "data_source": "gsm8k",
            "prompt": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": row["question"] + " Give a numerical answer inside \\boxed{}."},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {"question": row["question"]},
        }
    parquet_path = f"{WORK}/gsm8k_train500.parquet"
    ds.map(to_verl, remove_columns=ds.column_names).to_parquet(parquet_path)

    # ---- launch verl's Hydra CLI ----
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files=[{parquet_path}]",
        f"data.val_files=[{parquet_path}]",
        "data.train_batch_size=8",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        f"actor_rollout_ref.model.path={MODEL}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.optim.lr=1e-5",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.max_num_seqs=64",
        "actor_rollout_ref.rollout.response_length=512",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.kl_ctrl.type=fixed",
        "algorithm.kl_ctrl.kl_coef=0.01",
        "trainer.logger=[console,wandb]",
        f"trainer.project_name={PROJECT}",
        f"trainer.experiment_name={RUN_NAME}",
        "trainer.total_training_steps=30",
        "trainer.total_epochs=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        "trainer.seed=42",
    ]
    print("[verl] cmd:", " ".join(cmd))
    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    proc = subprocess.run(cmd, cwd=WORK, env=env)
    duration = time.time() - start
    print(f"[verl] subprocess exit={proc.returncode} in {duration:.1f}s")

    # ---- pull reward trace from W&B ----
    reward_trace = []
    wandb_url = None
    try:
        api = wandb.Api()
        runs = api.runs(
            f"{api.default_entity}/{PROJECT}",
            {"display_name": RUN_NAME},
        )
        if runs:
            wandb_url = runs[0].url
            hist = runs[0].history(
                keys=["critic/rewards/mean", "train/reward_mean", "reward/mean"],
            )
            for col in ("critic/rewards/mean", "train/reward_mean", "reward/mean"):
                if col in hist.columns:
                    reward_trace = [float(x) for x in hist[col].dropna().tolist()]
                    break
    except Exception as exc:
        print(f"[verl] could not fetch W&B history: {exc}")

    last10 = float(np.mean(reward_trace[-10:])) if len(reward_trace) >= 10 else float(np.mean(reward_trace or [0]))
    peak = float(max(reward_trace or [0]))
    first5 = float(np.mean(reward_trace[:5])) if reward_trace else 0.0

    return {
        "framework": "verl",
        "mode": "real",
        "model": MODEL,
        "algorithm": "GRPO",
        "seed": 42,
        "group_size": 8,
        "learning_rate": 1e-5,
        "steps": len(reward_trace),
        "task": "gsm8k-500",
        "platform": "modal-h100",
        "peak_reward": peak,
        "last10_avg": last10,
        "first5_avg": first5,
        "reward_trace": reward_trace,
        "duration_s": duration,
        "subprocess_exit": proc.returncode,
        "wandb_run_url": wandb_url,
    }


@app.local_entrypoint()
def main():
    print("Launching verl GRPO (Qwen3-8B, G=8, lr=1e-5, GSM8K-500, 30 steps) on H100...")
    result = run_verl_qwen3_8b.remote()
    os.makedirs("/home/user/workspace/elevation_outputs", exist_ok=True)
    out = "/home/user/workspace/elevation_outputs/modal_verl_grpo.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done. Wrote {out}.")
    print(f"peak={result.get('peak_reward')}  last10={result.get('last10_avg')}  steps={result.get('steps')}  exit={result.get('subprocess_exit')}")
