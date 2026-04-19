"""Modal GRPO Campaign — OpenRLHF launcher.

Runs the Task 4 framework-gap config on a single Modal H100 via pypi openrlhf:

    Qwen3-8B, seed=42, G=8 (n_samples_per_prompt=8), lr=1e-5, GSM8K[:500], 30 steps.

GRPO is realised via ``--advantage_estimator group_norm`` on the
``openrlhf.cli.train_ppo`` (non-Ray, single-node) entrypoint. A tiny local HTTP
reward server exposes the verifiable GSM8K boxed-answer reward so OpenRLHF can
query it per sample.

Sister script to ``experiments/modal/modal_grpo_trl.py`` and
``modal_grpo_verl.py``. Writes result.json to
``/home/user/workspace/elevation_outputs/modal_openrlhf_grpo.json``.

Note: the repo's top-level ``openrlhf/`` package shares a name with pypi
``openrlhf``. We avoid the shadow by not importing the repo launcher here and
calling pypi openrlhf via its CLI directly.
"""

import json
import os

import modal

app = modal.App("tinker-rl-openrlhf-grpo")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("packaging", "wheel", "ninja", "setuptools")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "pyarrow",
        "pydantic>=2",
        "pyyaml",
        "fastapi",
        "uvicorn",
    )
    .pip_install("flash-attn==2.7.0.post2", extra_options="--no-build-isolation")
    .pip_install(
        "vllm==0.7.2",
        "ray[default]>=2.34",
        "openrlhf==0.8.4",
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
def run_openrlhf_qwen3_8b():
    """Execute the OpenRLHF GRPO GSM8K-500 run on a single H100."""
    import re
    import subprocess
    import sys
    import threading
    import time

    from datasets import load_dataset
    from fastapi import FastAPI
    import numpy as np
    import uvicorn
    import wandb

    MODEL = "Qwen/Qwen3-8B"
    PROJECT = "tinker-rl-lab-world-class"
    RUN_NAME = "modal-openrlhf-qwen3-8b-grpo-gsm8k500"
    WORK = "/root/openrlhf-run"
    os.makedirs(WORK, exist_ok=True)

    # ---- GSM8K[:500] -> prompt + label JSONL ----
    SYS = "You are a math assistant. Solve step by step, then give your final answer inside \\boxed{}."
    ds = load_dataset("openai/gsm8k", "main", split="train[:500]")
    label_map = {}
    jsonl_path = f"{WORK}/gsm8k500.jsonl"
    with open(jsonl_path, "w") as f:
        for row in ds:
            m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            gt = m.group(1).replace(",", "").strip() if m else ""
            prompt = (
                f"<|im_start|>system\n{SYS}<|im_end|>\n"
                f"<|im_start|>user\n{row['question']} Give a numerical answer inside \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            label_map[prompt] = gt
            f.write(json.dumps({"prompt": prompt, "label": gt}) + "\n")

    # ---- reward server ----
    reward_app = FastAPI()

    def score(text, gt):
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text or "")
        for b in boxed:
            b_clean = b.strip().replace(",", "")
            try:
                if gt and abs(float(b_clean) - float(gt)) < 1e-2:
                    return 1.0
            except Exception:
                if b_clean == gt:
                    return 1.0
        return 0.0

    @reward_app.post("/get_reward")
    async def get_reward(payload: dict):
        queries = payload.get("query", [])
        labels = payload.get("label", [])
        rewards = [score(q, l) for q, l in zip(queries, labels)]
        return {"rewards": rewards}

    threading.Thread(
        target=lambda: uvicorn.run(reward_app, host="127.0.0.1", port=8765, log_level="warning"),
        daemon=True,
    ).start()
    time.sleep(5)

    # ---- launch OpenRLHF ----
    cmd = [
        sys.executable, "-m", "openrlhf.cli.train_ppo",
        "--pretrain", MODEL,
        "--prompt_data", jsonl_path,
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
        "--actor_learning_rate", "1e-5",
        "--init_kl_coef", "0.01",
        "--use_kl_loss",
        "--kl_estimator", "k3",
        "--save_steps", "-1",
        "--logging_steps", "1",
        "--max_samples", "240",
        "--save_path", f"{WORK}/ckpt",
        "--ckpt_path", f"{WORK}/ckpt",
        "--use_wandb", os.environ.get("WANDB_API_KEY", ""),
        "--wandb_project", PROJECT,
        "--wandb_run_name", RUN_NAME,
        "--bf16",
        "--flash_attn",
        "--remote_rm_url", "http://127.0.0.1:8765/get_reward",
        "--seed", "42",
    ]
    print("[openrlhf] cmd:", " ".join(cmd))
    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    proc = subprocess.run(cmd, cwd=WORK, env=env)
    duration = time.time() - start
    print(f"[openrlhf] subprocess exit={proc.returncode} in {duration:.1f}s")

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
                keys=["train/reward_mean", "reward/mean", "critic/rewards/mean"],
            )
            for col in ("train/reward_mean", "reward/mean", "critic/rewards/mean"):
                if col in hist.columns:
                    reward_trace = [float(x) for x in hist[col].dropna().tolist()]
                    break
    except Exception as exc:
        print(f"[openrlhf] could not fetch W&B history: {exc}")

    last10 = float(np.mean(reward_trace[-10:])) if len(reward_trace) >= 10 else float(np.mean(reward_trace or [0]))
    peak = float(max(reward_trace or [0]))
    first5 = float(np.mean(reward_trace[:5])) if reward_trace else 0.0

    return {
        "framework": "OpenRLHF",
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
    print("Launching OpenRLHF GRPO (Qwen3-8B, G=8, lr=1e-5, GSM8K-500, 30 steps) on H100...")
    result = run_openrlhf_qwen3_8b.remote()
    os.makedirs("/home/user/workspace/elevation_outputs", exist_ok=True)
    out = "/home/user/workspace/elevation_outputs/modal_openrlhf_grpo.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done. Wrote {out}.")
    print(f"peak={result.get('peak_reward')}  last10={result.get('last10_avg')}  steps={result.get('steps')}  exit={result.get('subprocess_exit')}")
