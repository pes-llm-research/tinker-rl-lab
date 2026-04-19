"""GRPO fine-tuning on GSM8K via Tinker SDK (Thinking Machines managed infra).

AI Scientist v2 template — Tinker-native. No local GPU required: the `tinker`
SDK ships training to TML's cluster. Local process orchestrates rollouts,
computes GRPO advantages, and drives optimizer steps.

Key variables an AI Scientist agent can modify:
- MODEL           — base model id (Tinker catalog)
- LORA_RANK       — LoRA adapter rank
- GROUP_SIZE      — GRPO group size (num samples per prompt)
- STEPS           — number of optimizer steps
- LR              — learning rate
- NUM_SEEDS       — seeds for reporting variance
- SYSTEM_PROMPT   — formatting instructions
- reward_fn       — extract answer, compute reward
- curriculum()    — problem selection strategy

Output: {out_dir}/final_info.json  with:
  {
    "gsm8k_training": {
      "means":     {last_10_accuracy_mean, peak_accuracy_mean, first_5_accuracy_mean,
                    training_loss_mean, duration_seconds_mean},
      "stderrs":   {...paired with _stderr},
      "final_info_dict": {...per-seed lists...}
    }
  }
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import tinker
import tinker.types as T

warnings.filterwarnings("ignore")

# ── Config (agent-editable) ──────────────────────────────────────────────
MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Tinker catalog. Small+fast.
LORA_RANK = 16
GROUP_SIZE = 4
STEPS = 30
LR = 5e-6
NUM_SEEDS = 3
PROMPTS_PER_STEP = 4
MAX_RESPONSE_TOKENS = 512
NUM_EVAL_PROBLEMS = 100

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, then give the final "
    "numerical answer inside \\boxed{}."
)

# ── Reward function (agent-editable) ─────────────────────────────────────
BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def extract_answer(text: str) -> str | None:
    m = BOXED.search(text)
    if not m:
        return None
    raw = m.group(1).strip().replace(",", "").replace("$", "")
    try:
        return str(int(float(raw)))
    except ValueError:
        return None


def reference_answer(row_answer: str) -> str | None:
    # GSM8K answers end with `#### <number>`
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", row_answer)
    if not m:
        return None
    return str(int(float(m.group(1).replace(",", ""))))


def reward_fn(response_text: str, reference: str) -> float:
    """Return a scalar reward. Default: binary exact match on \\boxed{} contents."""
    pred = extract_answer(response_text)
    if pred is None:
        return 0.0
    return 1.0 if pred == reference else 0.0


# ── Curriculum (agent-editable) ──────────────────────────────────────────
def curriculum(examples: list, step: int, total_steps: int) -> list:
    """Return the pool of examples to sample this step. Default: uniform."""
    return examples


# ── GRPO loss ────────────────────────────────────────────────────────────
_advantages: list[float] = []


def grpo_loss_fn(data, logprobs_list):
    losses = []
    for i, logprobs in enumerate(logprobs_list):
        adv = _advantages[i]
        losses.append(-adv * logprobs.sum())
    loss = torch.stack(losses).mean()
    return loss, {"grpo_loss": loss.item()}


# ── Training driver ──────────────────────────────────────────────────────
def run_one_seed(seed: int, examples_train: list, examples_eval: list) -> dict:
    global _advantages
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n── seed={seed} — connecting to Tinker ──")
    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
    print(f"model_id: {tc.model_id}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    start = time.time()

    w0 = tc.save_weights_for_sampler(name=f"seed{seed}_step0").result()
    sc = tc.create_sampling_client(model_path=w0.path)

    step_accuracy: list[float] = []
    step_loss: list[float] = []

    for step in range(STEPS):
        pool = curriculum(examples_train, step, STEPS)
        batch = random.sample(pool, min(PROMPTS_PER_STEP, len(pool)))

        all_data, all_advs, batch_rewards = [], [], []

        for prompt_text, reference in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tok.encode(rendered, add_special_tokens=False)
            prompt_mi = T.ModelInput.from_ints(prompt_ids)

            sp = T.SamplingParams(max_tokens=MAX_RESPONSE_TOKENS, temperature=0.8, top_p=0.95)
            responses = sc.sample(
                prompt_mi, num_samples=GROUP_SIZE, sampling_params=sp
            ).result()

            rewards = []
            for resp in responses.sequences:
                text = tok.decode(list(resp.tokens), skip_special_tokens=True)
                rewards.append(reward_fn(text, reference))

            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
            advs = [(r - mean_r) / std_r for r in rewards]
            batch_rewards.extend(rewards)

            for resp, adv in zip(responses.sequences, advs):
                resp_ids = list(resp.tokens)
                full_ids = prompt_ids + resp_ids
                target_ids = full_ids[1:] + [0]
                all_data.append(
                    T.Datum(
                        model_input=T.ModelInput.from_ints(full_ids),
                        loss_fn_inputs={
                            "target_tokens": T.TensorData(
                                data=target_ids, dtype="int64", shape=[len(target_ids)]
                            )
                        },
                    )
                )
                all_advs.append(adv)

        if not all_data:
            step_accuracy.append(0.0)
            step_loss.append(float("nan"))
            continue

        _advantages = all_advs
        result = tc.forward_backward_custom(
            data=all_data, loss_fn=grpo_loss_fn, loss_type_input="logprobs"
        ).result()
        tc.optim_step(
            T.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)
        ).result()

        acc = sum(batch_rewards) / len(batch_rewards)
        loss_val = result.metrics.get("grpo_loss", float("nan"))
        step_accuracy.append(acc)
        step_loss.append(loss_val)

        print(f"  step {step + 1:3d}/{STEPS} | loss={loss_val:.4f} | acc={acc:.3f}")

        # Refresh sampler every 5 steps so rollouts reflect updated policy
        if (step + 1) % 5 == 0 and step + 1 < STEPS:
            ckpt = tc.save_weights_for_sampler(name=f"seed{seed}_step{step + 1}").result()
            sc = tc.create_sampling_client(model_path=ckpt.path)

    duration = time.time() - start

    first_5 = float(np.mean(step_accuracy[:5])) if step_accuracy else 0.0
    last_10 = float(np.mean(step_accuracy[-10:])) if step_accuracy else 0.0
    peak = float(np.max(step_accuracy)) if step_accuracy else 0.0
    loss_mean = float(np.nanmean(step_loss)) if step_loss else float("nan")

    print(
        f"seed={seed} done | first5={first_5:.3f} last10={last_10:.3f} peak={peak:.3f} "
        f"loss={loss_mean:.3f} time={duration:.0f}s"
    )

    return {
        "first_5_accuracy": first_5,
        "last_10_accuracy": last_10,
        "peak_accuracy": peak,
        "training_loss": loss_mean,
        "duration_seconds": duration,
    }


# ── Main execution (global scope — runs immediately) ──────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

if "TINKER_API_KEY" not in os.environ:
    print("WARNING: TINKER_API_KEY not set. Skipping live training.")
    # Write a minimal final_info so the agent still sees output structure
    final_info = {
        "gsm8k_training": {
            "means": {
                "last_10_accuracy_mean": 0.0,
                "peak_accuracy_mean": 0.0,
                "first_5_accuracy_mean": 0.0,
                "training_loss_mean": float("nan"),
                "duration_seconds_mean": 0.0,
            },
            "stderrs": {
                "last_10_accuracy_stderr": 0.0,
                "peak_accuracy_stderr": 0.0,
                "first_5_accuracy_stderr": 0.0,
                "training_loss_stderr": 0.0,
                "duration_seconds_stderr": 0.0,
            },
            "final_info_dict": {
                "last_10_accuracy": [0.0],
                "peak_accuracy": [0.0],
                "first_5_accuracy": [0.0],
                "training_loss": [float("nan")],
                "duration_seconds": [0.0],
            },
            "config": {
                "model": MODEL,
                "lora_rank": LORA_RANK,
                "group_size": GROUP_SIZE,
                "steps": STEPS,
                "lr": LR,
                "num_seeds": NUM_SEEDS,
                "prompts_per_step": PROMPTS_PER_STEP,
            },
        }
    }
    # Save experiment_data.npy for AI Scientist plotting code
    experiment_data = {
        "gsm8k": {
            "metrics": {"train": [0.0], "val": [0.0]},
            "losses": {"train": [float("nan")], "val": [float("nan")]},
            "predictions": [],
            "ground_truth": [],
        }
    }
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print(f"Wrote placeholder experiment_data.npy")

    out = Path(working_dir) / "final_info.json"
    out.write_text(json.dumps(final_info, indent=2))
    print(f"Wrote placeholder {out}")
    print("\nMETRICS SUMMARY:")
    print("- Dataset: gsm8k (placeholder)")
    print("  - last_10_accuracy: final=0.000, best=0.000")
    print("  - training_loss: final=nan, best=nan")
    print("  - duration_seconds: final=0.000, best=0.000")
else:
    print("Loading GSM8K…")
    ds = load_dataset("openai/gsm8k", "main")
    train_rows = [
        (r["question"], reference_answer(r["answer"]))
        for r in ds["train"]
        if reference_answer(r["answer"]) is not None
    ]
    # Keep train modest; curriculum pools from this
    train_rows = train_rows[:500]
    examples_eval = [
        (r["question"], reference_answer(r["answer"]))
        for r in ds["test"]
        if reference_answer(r["answer"]) is not None
    ][:NUM_EVAL_PROBLEMS]
    print(f"  train: {len(train_rows)}  eval: {len(examples_eval)}")

    per_seed: list[dict] = []
    for seed in range(NUM_SEEDS):
        per_seed.append(run_one_seed(seed, train_rows, examples_eval))

    def stat(key: str):
        vals = [s[key] for s in per_seed]
        return {
            "mean": float(np.mean(vals)),
            "stderr": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "list": vals,
        }

    keys = [
        "last_10_accuracy",
        "peak_accuracy",
        "first_5_accuracy",
        "training_loss",
        "duration_seconds",
    ]
    stats = {k: stat(k) for k in keys}

    final_info = {
        "gsm8k_training": {
            "means": {f"{k}_mean": stats[k]["mean"] for k in keys},
            "stderrs": {f"{k}_stderr": stats[k]["stderr"] for k in keys},
            "final_info_dict": {k: stats[k]["list"] for k in keys},
            "config": {
                "model": MODEL,
                "lora_rank": LORA_RANK,
                "group_size": GROUP_SIZE,
                "steps": STEPS,
                "lr": LR,
                "num_seeds": NUM_SEEDS,
                "prompts_per_step": PROMPTS_PER_STEP,
            },
        }
    }
    # Save experiment_data.npy for AI Scientist plotting code
    experiment_data = {
        "gsm8k": {
            "metrics": {
                "train": [s["last_10_accuracy"] for s in per_seed],
                "val": [s["peak_accuracy"] for s in per_seed],
            },
            "losses": {
                "train": [s["training_loss"] for s in per_seed],
                "val": [s["training_loss"] for s in per_seed],
            },
            "predictions": [],
            "ground_truth": [],
        }
    }
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print(f"Wrote experiment_data.npy")

    out = Path(working_dir) / "final_info.json"
    out.write_text(json.dumps(final_info, indent=2))
    print(f"Wrote {out}")
    print(
        f"last_10={stats['last_10_accuracy']['mean']:.3f} ± "
        f"{stats['last_10_accuracy']['stderr']:.3f}  "
        f"peak={stats['peak_accuracy']['mean']:.3f}"
    )
    print("\nMETRICS SUMMARY:")
    print("- Dataset: gsm8k")
    for k in keys:
        mean_v = stats[k]["mean"]
        best_v = max(stats[k]["list"]) if k != "training_loss" else min(stats[k]["list"])
        print(f"  - {k}: final={mean_v:.4f}, best={best_v:.4f}")
