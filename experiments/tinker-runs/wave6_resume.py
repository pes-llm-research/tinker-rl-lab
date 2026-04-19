#!/usr/bin/env python3
"""Resume Wave 6 — rerun only the runs missing from wave6_ablations.json."""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wave6_ablations import (
    run_one, BASE_RANK, BASE_TEMP, BASE_BATCH, RESULTS_PATH,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# Load existing file
with open(RESULTS_PATH) as f:
    data = json.load(f)

existing = {r["tag"]: r for r in data.get("runs", [])}

# Build only failed/missing experiments
TO_RERUN = []

# Temperature sweep
for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
    tag = (f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}" if t == BASE_TEMP
           else f"w6_temp{t}_r{BASE_RANK}_b{BASE_BATCH}")
    if existing.get(tag, {}).get("status") != "completed":
        TO_RERUN.append({
            "tag": tag, "sweep": "temperature",
            "rank": BASE_RANK, "temperature": t, "batch": BASE_BATCH,
        })

# Rank sweep
for r in [4, 8, 16, 32, 64]:
    tag = (f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}" if r == BASE_RANK
           else f"w6_rank{r}_t{BASE_TEMP}_b{BASE_BATCH}")
    if existing.get(tag, {}).get("status") != "completed":
        TO_RERUN.append({
            "tag": tag, "sweep": "rank",
            "rank": r, "temperature": BASE_TEMP, "batch": BASE_BATCH,
        })

# Batch sweep
for b in [1, 2, 4, 8]:
    tag = (f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}" if b == BASE_BATCH
           else f"w6_batch{b}_r{BASE_RANK}_t{BASE_TEMP}")
    if existing.get(tag, {}).get("status") != "completed":
        TO_RERUN.append({
            "tag": tag, "sweep": "batch",
            "rank": BASE_RANK, "temperature": BASE_TEMP, "batch": b,
        })

print(f"Re-running {len(TO_RERUN)} experiments")
for e in TO_RERUN:
    print(f"  - {e['tag']} ({e['sweep']}) rank={e['rank']} temp={e['temperature']} batch={e['batch']}")

# Run
new_results = []
max_parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 6
with ThreadPoolExecutor(max_workers=max_parallel) as ex:
    futures = {ex.submit(run_one, e): e for e in TO_RERUN}
    for fut in as_completed(futures):
        try:
            new_results.append(fut.result())
        except Exception as e:
            exp = futures[fut]
            new_results.append({"tag": exp["tag"], "sweep": exp["sweep"],
                                "status": "crashed", "error": str(e)})

# Merge: replace failed entries with the new results (only successes win)
# Keep existing completed runs intact.
all_runs_by_tag = dict(existing)
for r in new_results:
    if r.get("status") == "completed" or r["tag"] not in all_runs_by_tag:
        all_runs_by_tag[r["tag"]] = r
    else:
        # keep existing (might have partial info) but don't clobber with crashed
        pass

merged_runs = list(all_runs_by_tag.values())


def _collect(sweep_name, key, values):
    rows = []
    for v in values:
        if sweep_name == "temperature":
            tag = (f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}"
                   if v == BASE_TEMP
                   else f"w6_temp{v}_r{BASE_RANK}_b{BASE_BATCH}")
        elif sweep_name == "rank":
            tag = (f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}"
                   if v == BASE_RANK
                   else f"w6_rank{v}_t{BASE_TEMP}_b{BASE_BATCH}")
        else:
            tag = (f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}"
                   if v == BASE_BATCH
                   else f"w6_batch{v}_r{BASE_RANK}_t{BASE_TEMP}")
        r = all_runs_by_tag.get(tag)
        if r:
            rows.append({key: v, **{k: r.get(k) for k in [
                "status", "peak_reward", "last10_avg", "first5_avg",
                "zero_reward_pct", "zero_loss_pct", "reward_trace",
                "step_log", "run_id", "checkpoint", "seed", "rank",
                "temperature", "batch", "group_size", "lr", "steps",
                "wall_clock_sec", "tag",
            ]}})
        else:
            rows.append({key: v, "status": "pending", "tag": tag})
    return rows


total = 13  # 5 temp + 5 rank + 4 batch - 2 baseline dups = 12, plus 1 baseline listing
completed = sum(1 for r in merged_runs if r.get("status") == "completed")
failed = sum(1 for r in merged_runs if r.get("status") in ("failed", "crashed"))

out = {
    "metadata": {
        "wave": 6,
        "title": "Qwen3-8B GSM8K Sensitivity Ablations",
        "model": "Qwen/Qwen3-8B",
        "task": "gsm8k",
        "seed": 42,
        "steps": 30,
        "group_size": 8,
        "lr": 1e-5,
        "baseline": {"rank": BASE_RANK, "temperature": BASE_TEMP, "batch": BASE_BATCH},
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(merged_runs),
        "completed": completed,
        "failed": failed,
        "finalized": True,
        "wandb_project": "https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class",
    },
    "temperature_sweep": _collect("temperature", "temperature",
                                  [0.2, 0.4, 0.6, 0.8, 1.0]),
    "rank_sweep": _collect("rank", "rank", [4, 8, 16, 32, 64]),
    "batch_sweep": _collect("batch", "batch", [1, 2, 4, 8]),
    "runs": merged_runs,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nDone. completed={completed} failed={failed}")
