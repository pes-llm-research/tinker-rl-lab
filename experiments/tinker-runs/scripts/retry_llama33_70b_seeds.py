#!/usr/bin/env python3
"""
Retry only the failed seeds from Task 1 (Llama-3.3-70B-Instruct multi-seed).

Reads existing llama33_70b_seeds.json, retries any seed whose status != 'completed',
and merges the new results back in place (keeping the earlier completed runs).
"""
import os, sys, json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = "/home/user/workspace/tinker-rl-lab"
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "tinker-runs"))

from campaign_v2 import run_experiment  # noqa: E402

OUT_PATH = os.path.join(
    REPO_ROOT, "experiments", "tinker-runs", "results", "llama33_70b_seeds.json"
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
GROUP_SIZE = 8
LR = 1e-5
STEPS = 30
WAVE = "task1-llama33-70b-multiseed"
ALL_SEEDS = [42, 123, 456, 789]


def main(max_parallel: int = 2):
    with open(OUT_PATH) as f:
        payload = json.load(f)

    results = list(payload.get("results", []))

    # Determine which seeds still need a successful run.
    completed_seeds = {r["seed"] for r in results if r.get("status") == "completed"}
    retry_seeds = [s for s in ALL_SEEDS if s not in completed_seeds]
    if not retry_seeds:
        print("All seeds already completed — nothing to retry.")
        return

    print(f"Retrying seeds: {retry_seeds}")

    # Drop old failed attempts for these seeds so we don't carry stale errors.
    results = [r for r in results if r.get("seed") not in retry_seeds]

    exps = [{
        "tag": f"w1_llama33-70b-inst_s{seed}",
        "model": MODEL, "wave": WAVE, "priority": "HIGH",
        "steps": STEPS, "group_size": GROUP_SIZE, "lr": LR, "seed": seed,
    } for seed in retry_seeds]

    def snapshot():
        payload["results"] = results
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        payload["completed"] = sum(1 for r in results if r.get("status") == "completed")
        payload["failed"] = sum(1 for r in results if r.get("status") == "failed")
        payload["total"] = len(ALL_SEEDS)
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
            except Exception as e:
                r = {"tag": exp["tag"], "model": exp["model"], "wave": exp["wave"],
                     "status": "failed", "error": f"Executor error: {e}",
                     "timestamp": datetime.utcnow().isoformat()}
            results.append(r)
            snapshot()
            print(f"  -> {r.get('tag')} status={r.get('status')} "
                  f"peak={r.get('peak_reward')} last10={r.get('last10_avg')}")

    # Sort final list by seed for stable diffs.
    results.sort(key=lambda r: (r.get("seed") or 0))
    snapshot()
    print(f"\nUpdated {OUT_PATH}")


if __name__ == "__main__":
    mp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(max_parallel=mp)
