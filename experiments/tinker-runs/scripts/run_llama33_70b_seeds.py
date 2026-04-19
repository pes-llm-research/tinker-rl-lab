#!/usr/bin/env python3
"""
Task 1 — Llama-3.3-70B-Instruct multi-seed retry on GSM8K.

Reuses the run_experiment() + EXPERIMENT schema from
experiments/tinker-runs/campaign_v2.py so behaviour is identical to
`w1_llama33-70b-inst` from that campaign, just across seeds {42, 123, 456, 789}.

Output: experiments/tinker-runs/results/llama33_70b_seeds.json
Then: regenerate master_results.json via experiments/aggregate_results.py
"""
import os, sys, json, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = "/home/user/workspace/tinker-rl-lab"
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "tinker-runs"))

# Import the shared experiment runner to avoid code duplication.
from campaign_v2 import run_experiment  # noqa: E402

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
SEEDS = [42, 123, 456, 789]
GROUP_SIZE = 8
LR = 1e-5
STEPS = 30
WAVE = "task1-llama33-70b-multiseed"

OUT_PATH = os.path.join(
    REPO_ROOT, "experiments", "tinker-runs", "results", "llama33_70b_seeds.json"
)


def build_exps():
    exps = []
    for seed in SEEDS:
        exps.append({
            "tag": f"w1_llama33-70b-inst_s{seed}",
            "model": MODEL,
            "wave": WAVE,
            "priority": "HIGH",
            "steps": STEPS,
            "group_size": GROUP_SIZE,
            "lr": LR,
            "seed": seed,
        })
    return exps


def main(max_parallel: int = 2):
    exps = build_exps()
    print(f"Launching {len(exps)} Llama-3.3-70B-Instruct runs "
          f"(G={GROUP_SIZE}, lr={LR}, steps={STEPS}) @ max_parallel={max_parallel}")

    results = []
    started = datetime.utcnow().isoformat() + "Z"

    def snapshot():
        payload = {
            "task": "task1_llama33_70b_multiseed",
            "model": MODEL,
            "wave": WAVE,
            "seeds": SEEDS,
            "group_size": GROUP_SIZE,
            "lr": LR,
            "steps": STEPS,
            "started_at": started,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "completed": sum(1 for r in results if r.get("status") == "completed"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "total": len(exps),
            "results": results,
        }
        tmp = OUT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, OUT_PATH)

    snapshot()

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(run_experiment, e): e for e in exps}
        for fut in as_completed(futs):
            exp = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # pragma: no cover — defensive
                r = {"tag": exp["tag"], "model": exp["model"], "wave": exp["wave"],
                     "status": "failed", "error": f"Executor error: {e}",
                     "timestamp": datetime.utcnow().isoformat()}
            results.append(r)
            snapshot()
            print(f"  -> {r.get('tag')} status={r.get('status')} "
                  f"peak={r.get('peak_reward')} last10={r.get('last10_avg')}")

    print(f"\nSaved to {OUT_PATH}")
    print(f"  completed={sum(1 for r in results if r.get('status')=='completed')} "
          f"failed={sum(1 for r in results if r.get('status')=='failed')}")


if __name__ == "__main__":
    mp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(max_parallel=mp)
