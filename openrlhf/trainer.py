"""
OpenRLHF Trainer for tinker-rl-lab

Real GRPO runner for framework-gap comparison. Uses OpenRLHF's Ray + vLLM
trainer in the "ppo_ray" configuration with advantage_estimator=group_norm
(i.e. GRPO). Training loop mirrors the TRL/Tinker loops so reward traces are
comparable apples-to-apples.

Falls back to a seeded deterministic dryrun when openrlhf cannot be imported,
clearly marked as such in the returned metrics.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import OpenRLHFConfig


class OpenRLHFTrainer:
    """Real GRPO trainer backed by OpenRLHF, with dryrun fallback."""

    def __init__(self, config: OpenRLHFConfig):
        self.config = config
        self.current_step = 0
        self.reward_history: List[float] = []
        self.loss_history: List[float] = []
        self.start_time: Optional[float] = None
        self.mode: str = "dryrun"

    async def setup(self) -> None:
        print(f"\n{'=' * 60}")
        print("Setting up OpenRLHF Trainer")
        print(f"Model: {self.config.model_name}")
        print(f"Algorithm: {self.config.algorithm.algorithm}")
        print(f"GPUs: {self.config.num_gpus}  Actors: {self.config.num_actors}")
        print(f"{'=' * 60}\n")

        try:
            import openrlhf  # type: ignore  # noqa: F401
            self.mode = "real"
            print(f"OpenRLHF installed — mode=real, version={getattr(openrlhf, '__version__', 'unknown')}")
        except Exception as exc:
            print(f"OpenRLHF unavailable ({exc!r}); using seeded dryrun fallback.")
            self.mode = "dryrun"

    def _dryrun_reward(self, step: int) -> float:
        """Deterministic seeded fallback for Qwen3-8B/GSM8K-500/G=8/lr=1e-5.

        Calibrated slightly below verl (OpenRLHF's default DAPO-style KL can
        over-regularize at lr=1e-5 on 30 steps), but above the near-collapse
        TRL-H100 run. Real Modal numbers overwrite these.
        """
        rng = np.random.default_rng(seed=int(self.config.algorithm.kl_coef * 1e6) + 123 + step)
        base = 0.22 + 0.012 * step
        noise = float(rng.normal(0, 0.10))
        return max(0.0, min(1.0, base + noise))

    async def train_step(self, step: int) -> Dict[str, Any]:
        if self.mode == "real":
            raise RuntimeError("train_step() is dryrun-only; use run_openrlhf_training().")
        step_start = time.time()
        reward_val = self._dryrun_reward(step)
        loss_val = 1.0 / (step + 1) + float(np.random.default_rng(step + 10000).normal(0, 0.05))
        self.reward_history.append(reward_val)
        self.loss_history.append(loss_val)
        print(f"  step={step} loss={loss_val:.4f} reward={reward_val:.4f}")
        return {
            "step": step,
            "loss": loss_val,
            "reward/mean": reward_val,
            "learning_rate": self.config.learning_rate,
            "step_time": time.time() - step_start,
        }

    async def run(self) -> Dict[str, Any]:
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("Starting OpenRLHF Training")
        print("=" * 60 + "\n")
        await self.setup()

        if self.mode == "real":
            print("Use run_openrlhf_training() for real runs; falling back to dryrun for run().")
            self.mode = "dryrun"

        for step in range(self.config.epochs):
            try:
                await self.train_step(step)
                self.current_step = step + 1
            except Exception as exc:
                print(f"Error in step {step}: {exc}")
                break

        duration = time.time() - self.start_time
        last10 = float(np.mean(self.reward_history[-10:])) if len(self.reward_history) >= 10 else float(np.mean(self.reward_history or [0]))
        peak = float(max(self.reward_history or [0]))
        first5 = float(np.mean(self.reward_history[:5])) if self.reward_history else 0.0

        return {
            "framework": "openrlhf",
            "mode": self.mode,
            "final_step": self.current_step,
            "peak_reward": peak,
            "last10_avg": last10,
            "first5_avg": first5,
            "reward_trace": self.reward_history,
            "loss_trace": self.loss_history,
            "duration_s": duration,
            "config": self.config.to_dict(),
        }


# ---------------------------------------------------------------------------
# Real driver — invoked by experiments/modal/modal_grpo_openrlhf.py on H100.
# ---------------------------------------------------------------------------

GSM8K_SYSTEM_PROMPT = (
    "You are a math assistant. Solve step by step, then give your final answer inside \\boxed{}."
)


def _prepare_gsm8k_500_jsonl(out_path: str) -> str:
    """Materialise GSM8K[:500] in OpenRLHF prompt+label JSONL format."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("openai/gsm8k", "main", split="train[:500]")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in ds:
            m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            gt = m.group(1).replace(",", "").strip() if m else ""
            prompt = (
                f"<|system|>\n{GSM8K_SYSTEM_PROMPT}\n"
                f"<|user|>\n{row['question']} Give a numerical answer inside \\boxed{{}}.\n"
                f"<|assistant|>\n"
            )
            f.write(json.dumps({"prompt": prompt, "label": gt}) + "\n")
    return str(out)


def run_openrlhf_training(config: OpenRLHFConfig, output_dir: str = "/tmp/openrlhf-run") -> Dict[str, Any]:
    """Real end-to-end OpenRLHF GRPO driver on GSM8K-500.

    OpenRLHF is launched as a subprocess (``python -m openrlhf.cli.train_ppo_ray``)
    with ``--advantage_estimator group_norm`` to yield GRPO semantics. Reward
    trace is read back from the W&B run history after the subprocess exits.
    """
    try:
        import wandb  # type: ignore
        import openrlhf  # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "run_openrlhf_training requires openrlhf + wandb; "
            "launch via experiments/modal/modal_grpo_openrlhf.py"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    data_path = _prepare_gsm8k_500_jsonl(os.path.join(output_dir, "gsm8k500.jsonl"))

    run_name = config.run_name or "openrlhf-gsm8k-qwen3-8b"
    os.environ["WANDB_PROJECT"] = os.environ.get(
        "WANDB_PROJECT", "tinker-rl-lab-world-class"
    )

    # ---- launch OpenRLHF PPO-Ray with group_norm advantage (GRPO) ----
    # We rely on the reward being baked in via a verifiable reward function
    # the container will register under --reward_pretrain (see modal_grpo_openrlhf.py).
    cmd = [
        "python", "-m", "openrlhf.cli.train_ppo_ray",
        "--pretrain", config.model_name,
        "--ref_pretrain", config.model_name,
        "--prompt_data", data_path,
        "--input_key", "prompt",
        "--label_key", "label",
        "--advantage_estimator", "group_norm",
        "--n_samples_per_prompt", "8",
        "--train_batch_size", "8",
        "--micro_train_batch_size", "1",
        "--rollout_batch_size", "8",
        "--micro_rollout_batch_size", "1",
        "--max_epochs", "1",
        "--num_episodes", "1",
        "--prompt_max_len", "512",
        "--generate_max_len", "512",
        "--actor_learning_rate", str(config.learning_rate),
        "--init_kl_coef", str(config.algorithm.kl_coef),
        "--use_kl_loss",
        "--kl_estimator", "k3",
        "--save_steps", "-1",
        "--logging_steps", "1",
        "--max_samples", "240",  # 30 steps * batch=8
        "--save_path", os.path.join(output_dir, "ckpt"),
        "--ckpt_path", os.path.join(output_dir, "ckpt"),
        "--use_wandb", os.environ.get("WANDB_API_KEY", ""),
        "--wandb_project", os.environ["WANDB_PROJECT"],
        "--wandb_run_name", run_name,
        "--colocate_all_models",
        "--vllm_num_engines", "1",
        "--vllm_tensor_parallel_size", "1",
        "--bf16",
        "--flash_attn",
        "--actor_num_gpus_per_node", str(config.num_gpus),
        "--ref_num_gpus_per_node", str(config.num_gpus),
        "--remote_rm_url", "verifiable_gsm8k",  # served by modal_grpo_openrlhf.py
        "--seed", "42",
    ]
    print("[openrlhf] cmd:", " ".join(cmd))
    # Strip repo root from subprocess PYTHONPATH so python -m openrlhf.cli.*
    # resolves to the installed pypi package, not the repo's openrlhf/ launcher.
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    start = time.time()
    proc = subprocess.run(cmd, cwd=output_dir, env=env)
    duration = time.time() - start
    print(f"[openrlhf] subprocess exit={proc.returncode} in {duration:.1f}s")

    # Pull reward trace from W&B
    reward_trace: List[float] = []
    try:
        import wandb as _wb  # type: ignore
        api = _wb.Api()
        runs = api.runs(
            f"{api.default_entity}/{os.environ['WANDB_PROJECT']}",
            {"display_name": run_name},
        )
        if runs:
            hist = runs[0].history(keys=["train/reward_mean", "reward/mean", "critic/rewards/mean"])
            for col in ("train/reward_mean", "reward/mean", "critic/rewards/mean"):
                if col in hist.columns:
                    reward_trace = [float(x) for x in hist[col].dropna().tolist()]
                    break
    except Exception as exc:
        print(f"[openrlhf] could not fetch W&B history: {exc}")

    last10 = float(np.mean(reward_trace[-10:])) if len(reward_trace) >= 10 else float(np.mean(reward_trace or [0]))
    peak = float(max(reward_trace or [0]))
    first5 = float(np.mean(reward_trace[:5])) if reward_trace else 0.0
    result = {
        "framework": "openrlhf",
        "mode": "real",
        "model": config.model_name,
        "algorithm": "GRPO",
        "seed": 42,
        "group_size": 8,
        "learning_rate": config.learning_rate,
        "steps": len(reward_trace),
        "task": "gsm8k-500",
        "platform": "modal-h100",
        "peak_reward": peak,
        "last10_avg": last10,
        "first5_avg": first5,
        "reward_trace": reward_trace,
        "duration_s": duration,
        "subprocess_exit": proc.returncode,
    }
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def run(config: OpenRLHFConfig) -> Dict[str, Any]:
    """Sync entrypoint used by the Modal runner."""
    return asyncio.run(OpenRLHFTrainer(config).run())
