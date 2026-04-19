#!/usr/bin/env python3
"""Upload Tinker training logs to W&B as historical runs."""

import re
import os
import wandb

LOGS_DIR = "experiments/tinker-runs/logs"
PROJECT = "tinker-rl-scaling"
GROUP = "gsm8k-grpo-qwen3-8b"

SEED_LOGS = {
    "042": "gsm8k_8B_s042.log",
    "137": "gsm8k_8B_s137.log",
    "256": "gsm8k_8B_s256.log",
    "512": "gsm8k_8B_s512.log",
    "999": "gsm8k_8B_s999.log",
}

# Also upload experiment ablation logs
ABLATION_LOGS = {
    "exp_a_baseline": "grpo_exp_a.log",
    "exp_b_high_lr": "grpo_exp_b.log",
    "exp_c_low_temp": "grpo_exp_c.log",
    "exp_d_xlam": "grpo_exp_d.log",
    "gsm8k_100step": "gsm8k_8B_100step.log",
    "gsm8k_rank8": "gsm8k_8B_rank8.log",
    "gsm8k_rank16": "gsm8k_8B_rank16.log",
    "gsm8k_rank64": "gsm8k_8B_rank64.log",
    "gsm8k_4B_s137": "gsm8k_4B_s137.log",
    "grpo_100_math": "grpo_100_math.log",
    "grpo_100_synth": "grpo_100_synth.log",
    "grpo_100_xlam": "grpo_100_xlam.log",
}

STEP_RE = re.compile(
    r"\s*(\d+)/(\d+)\s*\|\s*loss=([\-\d.]+)\s*\|\s*reward=([\d.]+)\s*\|\s*acc=([\d.]+)%"
)
RUN_ID_RE = re.compile(r"Run ID:\s*(\S+)")
RUN_RE = re.compile(r"Run:\s*(\S+)")
CONFIG_RE = re.compile(r"model=(\S+)\s+seed=(\d+)\s+rank=(\d+)\s+lr=([\d.e\-]+)")
REWARD_TRACE_RE = re.compile(r"Reward trace:\s*\[(.+)\]")


def parse_log(path: str):
    steps = []
    run_id = None
    config = {}
    reward_trace = None

    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(
                    {
                        "step": int(m.group(1)),
                        "total_steps": int(m.group(2)),
                        "loss": float(m.group(3)),
                        "reward": float(m.group(4)),
                        "accuracy": float(m.group(5)),
                    }
                )
                continue

            m = RUN_ID_RE.search(line)
            if m:
                run_id = m.group(1)
                continue

            if not run_id:
                m = RUN_RE.search(line)
                if m:
                    run_id = m.group(1)

            m = CONFIG_RE.search(line)
            if m:
                config = {
                    "model": m.group(1),
                    "seed": int(m.group(2)),
                    "lora_rank": int(m.group(3)),
                    "learning_rate": float(m.group(4)),
                }

            m = REWARD_TRACE_RE.search(line)
            if m:
                reward_trace = [float(x.strip()) for x in m.group(1).split(",")]

    return steps, run_id, config, reward_trace


def upload_run(name: str, log_file: str, group: str):
    path = os.path.join(LOGS_DIR, log_file)
    if not os.path.exists(path):
        print(f"  SKIP {log_file} (not found)")
        return

    steps, run_id, config, reward_trace = parse_log(path)
    if not steps:
        print(f"  SKIP {log_file} (no step data)")
        return

    config["tinker_run_id"] = run_id or "unknown"
    config["algorithm"] = "grpo"
    config["environment"] = "gsm8k"
    config["source"] = "tinker-api"

    run = wandb.init(
        project=PROJECT,
        group=group,
        name=name,
        config=config,
        reinit=True,
    )

    for s in steps:
        run.log(
            {
                "train/step": s["step"],
                "train/loss": s["loss"],
                "train/reward_mean": s["reward"],
                "train/accuracy": s["accuracy"],
            },
            step=s["step"],
        )

    # Summary
    if steps:
        last10 = [s["reward"] for s in steps[-10:]]
        run.summary["final_accuracy"] = steps[-1]["accuracy"]
        run.summary["final_reward"] = steps[-1]["reward"]
        run.summary["peak_accuracy"] = max(s["accuracy"] for s in steps)
        run.summary["mean_last10_reward"] = sum(last10) / len(last10)
        run.summary["total_steps"] = len(steps)
        run.summary["tinker_run_id"] = run_id

    run.finish()
    print(f"  OK {name}: {len(steps)} steps, final_acc={steps[-1]['accuracy']:.1f}%")


def main():
    print("=== Uploading Tinker GSM8K runs (5 seeds) ===")
    for seed, log_file in SEED_LOGS.items():
        name = f"gsm8k-qwen3-8b-seed{seed}-tinker"
        upload_run(name, log_file, group=GROUP)

    print("\n=== Uploading ablation/experiment runs ===")
    for name, log_file in ABLATION_LOGS.items():
        upload_run(f"{name}-tinker", log_file, group="gsm8k-ablations")

    print("\nDone! Check https://wandb.ai/ for project: tinker-rl-scaling")


if __name__ == "__main__":
    main()
