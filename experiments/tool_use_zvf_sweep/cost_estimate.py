#!/usr/bin/env python3
"""
Pre-flight cost estimator for one Tinker sweep run.

Computes an upper-bound token count and dollar estimate from a config YAML.
Rates are set from TINKER_RATE_SAMPLE_PER_M and TINKER_RATE_TRAIN_PER_M env
vars (USD per 1M tokens). If unset, the estimator aborts rather than
silently pretending cost is zero.

Usage:
  .venv/bin/python experiments/tool_use_zvf_sweep/cost_estimate.py <config.yaml>
  # exits 0 + prints JSON line. Nonzero exit means misconfigured/refuse.

Env knobs (set conservative upper-bound rates):
  TINKER_RATE_SAMPLE_PER_M   USD per 1M sampled output tokens
  TINKER_RATE_TRAIN_PER_M    USD per 1M trained tokens (LoRA update)
  TINKER_MODEL_MULT_<MODEL>  optional model multiplier (e.g. for 8B vs 0.5B)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import yaml


def _rate(name: str) -> float:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"ERROR: {name} not set. Export conservative rates before estimating, e.g.\n"
            f"  export TINKER_RATE_SAMPLE_PER_M=0.5   # USD / 1M sampled tokens\n"
            f"  export TINKER_RATE_TRAIN_PER_M=1.0    # USD / 1M trained tokens"
        )
    try:
        return float(v)
    except ValueError:
        raise SystemExit(f"ERROR: {name}={v!r} is not a number")


def estimate(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    env = cfg["env"]
    tinker = cfg["tinker"]

    total_steps = int(env["total_steps"])
    batch_size = int(env["batch_size"])
    group_size = int(env["group_size"])
    max_env_tok = int(env.get("max_token_length", 512))
    max_train_tok = int(tinker.get("max_token_trainer_length", 1024))
    tokenizer = env.get("tokenizer_name", "?")

    # Upper bounds: assume every sampled response is full length, every trained
    # example is full length, and evals run at steps_per_eval granularity.
    steps_per_eval = int(env.get("steps_per_eval", total_steps))
    eval_sets = max(1, total_steps // max(1, steps_per_eval))
    num_eval_reqs = int(cfg.get("openai", [{}])[0].get("num_requests_for_eval", 0))

    sample_tokens_train = total_steps * batch_size * group_size * max_env_tok
    sample_tokens_eval = eval_sets * num_eval_reqs * max_env_tok
    sample_tokens = sample_tokens_train + sample_tokens_eval
    train_tokens = total_steps * batch_size * max_train_tok

    sample_rate = _rate("TINKER_RATE_SAMPLE_PER_M")
    train_rate = _rate("TINKER_RATE_TRAIN_PER_M")

    # Optional model multiplier (e.g. for 8B vs 0.5B pricing)
    model_key = tokenizer.replace("/", "_").replace("-", "_").upper()
    mult_env = f"TINKER_MODEL_MULT_{model_key}"
    mult = float(os.environ.get(mult_env, "1.0"))

    sample_usd = (sample_tokens / 1_000_000) * sample_rate * mult
    train_usd = (train_tokens / 1_000_000) * train_rate * mult
    total_usd = sample_usd + train_usd

    return {
        "config": str(path),
        "tokenizer": tokenizer,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "group_size": group_size,
        "sample_tokens_upper": sample_tokens,
        "train_tokens_upper": train_tokens,
        "sample_usd_upper": round(sample_usd, 2),
        "train_usd_upper": round(train_usd, 2),
        "model_mult": mult,
        "usd_upper": round(total_usd, 2),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: cost_estimate.py <config.yaml> [more.yaml ...]")
    rows = [estimate(Path(p)) for p in sys.argv[1:]]
    total = round(sum(r["usd_upper"] for r in rows), 2)
    for r in rows:
        print(json.dumps(r))
    print(json.dumps({"_total_usd_upper": total, "_runs": len(rows)}))


if __name__ == "__main__":
    main()
