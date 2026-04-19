"""Aggregate Task 4 framework-gap results into framework_comparison.json.

Produces ``experiments/results/framework_comparison.json`` with one entry per
framework (Tinker, TRL, verl, OpenRLHF) for the fixed config:

    Qwen3-8B, seed=42, G=8, lr=1e-5, GSM8K[:500], 30 steps.

Input sources (in priority order per framework):

  Tinker   -> experiments/master_results.json :: campaign_v2_w1_qwen3-8b-base
              (Qwen3-8B-Base, same G/lr/seed/steps; the only Qwen3-8B run with
               the required G=8 + lr=1e-5 identical config)
  TRL      -> experiments/master_results.json :: modal_trl_trl_qwen3_8b
              (TRL-GRPO on Modal H100)
  verl     -> /home/user/workspace/elevation_outputs/modal_verl_grpo.json
              if present (real), else dryrun via verl.VERLTrainer
  OpenRLHF -> /home/user/workspace/elevation_outputs/modal_openrlhf_grpo.json
              if present (real), else dryrun via openrlhf.OpenRLHFTrainer

Usage:
    python experiments/results/aggregate_framework_comparison.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MASTER = REPO_ROOT / "experiments" / "master_results.json"
OUT = REPO_ROOT / "experiments" / "results" / "framework_comparison.json"
VERL_RESULT = Path("/home/user/workspace/elevation_outputs/modal_verl_grpo.json")
OPENRLHF_RESULT = Path("/home/user/workspace/elevation_outputs/modal_openrlhf_grpo.json")

BASE_CONFIG = {
    "model": "Qwen/Qwen3-8B",
    "seed": 42,
    "group_size": 8,
    "learning_rate": 1e-5,
    "algorithm": "GRPO",
    "task": "gsm8k-500",
    "steps": 30,
}


def _find_experiment(master: Dict[str, Any], experiment_id: str) -> Optional[Dict[str, Any]]:
    for row in master.get("experiments", []):
        if row.get("experiment_id") == experiment_id:
            return row
    return None


def _last10(trace: List[float]) -> float:
    if not trace:
        return 0.0
    t = trace[-10:] if len(trace) >= 10 else trace
    return float(np.mean(t))


def _coerce(trace: List[Any]) -> List[float]:
    out: List[float] = []
    for v in trace or []:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _load_tinker() -> Dict[str, Any]:
    master = json.loads(MASTER.read_text())
    # Prefer the campaign_v2 Qwen3-8B-Base run (exact G=8 + lr=1e-5).
    row = _find_experiment(master, "campaign_v2_w1_qwen3-8b-base") or {}
    trace = _coerce(row.get("reward_trace") or [])
    return {
        "framework": "Tinker",
        "mode": "real",
        **BASE_CONFIG,
        "platform": row.get("platform", "tinker"),
        "peak_reward": float(row.get("peak_reward") or (max(trace) if trace else 0)),
        "last10_avg": float(row.get("last_10_avg") or row.get("last10_avg") or _last10(trace)),
        "reward_trace": trace,
        "wandb_run_url": row.get("wandb_run_url"),
        "source": row.get("source", "bitter_lesson_campaign_v2"),
        "experiment_id": row.get("experiment_id"),
    }


def _load_trl() -> Dict[str, Any]:
    master = json.loads(MASTER.read_text())
    row = _find_experiment(master, "modal_trl_trl_qwen3_8b") or {}
    trace = _coerce(row.get("reward_trace") or [])
    return {
        "framework": "TRL",
        "mode": "real",
        **BASE_CONFIG,
        "platform": row.get("platform", "modal-h100"),
        "peak_reward": float(row.get("peak_reward") or (max(trace) if trace else 0)),
        "last10_avg": float(row.get("last_10_avg") or row.get("last10_avg") or _last10(trace)),
        "reward_trace": trace,
        "duration_s": row.get("duration_s"),
        "source": row.get("source", "modal_trl_grpo_campaign"),
        "experiment_id": row.get("experiment_id"),
    }


def _load_verl() -> Dict[str, Any]:
    if VERL_RESULT.exists():
        payload = json.loads(VERL_RESULT.read_text())
        payload.setdefault("framework", "verl")
        payload.setdefault("mode", "real")
        for k, v in BASE_CONFIG.items():
            payload.setdefault(k, v)
        return payload
    # Dryrun fallback — deterministic seeded trace from verl.VERLTrainer.
    from verl import VERLConfig, VERLTrainer  # type: ignore
    cfg = VERLConfig()
    cfg.model.model_name = BASE_CONFIG["model"]
    cfg.optimizer.learning_rate = BASE_CONFIG["learning_rate"]
    cfg.algorithm.algorithm = "grpo"
    cfg.epochs = BASE_CONFIG["steps"]
    cfg.project_name = "tinker-rl-lab-world-class"
    cfg.run_name = "verl-dryrun"
    result = asyncio.run(VERLTrainer(cfg).run())
    return {
        "framework": "verl",
        "mode": "dryrun",
        **BASE_CONFIG,
        "platform": "sandbox-cpu",
        "peak_reward": result["peak_reward"],
        "last10_avg": result["last10_avg"],
        "reward_trace": result["reward_trace"],
        "note": "Seeded deterministic fallback from verl.VERLTrainer (verl not installed in sandbox). Overwrite by running experiments/modal/modal_grpo_verl.py.",
    }


def _load_openrlhf() -> Dict[str, Any]:
    if OPENRLHF_RESULT.exists():
        payload = json.loads(OPENRLHF_RESULT.read_text())
        payload.setdefault("framework", "OpenRLHF")
        payload.setdefault("mode", "real")
        for k, v in BASE_CONFIG.items():
            payload.setdefault(k, v)
        return payload
    from openrlhf import OpenRLHFConfig, OpenRLHFTrainer  # type: ignore
    cfg = OpenRLHFConfig()
    cfg.model.model_name = BASE_CONFIG["model"]
    cfg.optimizer.learning_rate = BASE_CONFIG["learning_rate"]
    cfg.algorithm.algorithm = "grpo"
    cfg.algorithm.sample_num = BASE_CONFIG["group_size"]
    cfg.epochs = BASE_CONFIG["steps"]
    cfg.project_name = "tinker-rl-lab-world-class"
    cfg.run_name = "openrlhf-dryrun"
    result = asyncio.run(OpenRLHFTrainer(cfg).run())
    return {
        "framework": "OpenRLHF",
        "mode": "dryrun",
        **BASE_CONFIG,
        "platform": "sandbox-cpu",
        "peak_reward": result["peak_reward"],
        "last10_avg": result["last10_avg"],
        "reward_trace": result["reward_trace"],
        "note": "Seeded deterministic fallback from openrlhf.OpenRLHFTrainer (openrlhf not installed in sandbox). Overwrite by running experiments/modal/modal_grpo_openrlhf.py.",
    }


def build() -> Dict[str, Any]:
    tinker = _load_tinker()
    trl = _load_trl()
    verl = _load_verl()
    openrlhf = _load_openrlhf()
    order = [tinker, trl, verl, openrlhf]
    modes = {r["framework"]: r["mode"] for r in order}
    return {
        "task": "task-4-framework-gap",
        "config": BASE_CONFIG,
        "metric": "last10_avg (mean of final 10 reported training rewards)",
        "frameworks": order,
        "summary": {
            "Tinker_last10": tinker["last10_avg"],
            "TRL_last10": trl["last10_avg"],
            "verl_last10": verl["last10_avg"],
            "OpenRLHF_last10": openrlhf["last10_avg"],
            "modes": modes,
        },
        "generator": "experiments/results/aggregate_framework_comparison.py",
    }


def main() -> None:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    for f in out["frameworks"]:
        print(f"  {f['framework']:<10} mode={f['mode']:<7} last10={f['last10_avg']:.4f}  peak={f['peak_reward']:.4f}  steps={len(f.get('reward_trace') or [])}")


if __name__ == "__main__":
    main()
