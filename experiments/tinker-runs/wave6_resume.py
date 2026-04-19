#!/usr/bin/env python3
"""Resume Wave 6 — rerun only the runs missing from wave6_ablations.json."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone


def _compute_rerun_plan(existing, base_rank, base_temp, base_batch):
    """Return list of experiments whose tags are not yet completed."""
    to_rerun = []

    for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
        tag = (
            f"w6_base_r{base_rank}_t{base_temp}_b{base_batch}"
            if t == base_temp
            else f"w6_temp{t}_r{base_rank}_b{base_batch}"
        )
        if existing.get(tag, {}).get("status") != "completed":
            to_rerun.append(
                {
                    "tag": tag,
                    "sweep": "temperature",
                    "rank": base_rank,
                    "temperature": t,
                    "batch": base_batch,
                }
            )

    for r in [4, 8, 16, 32, 64]:
        tag = (
            f"w6_base_r{base_rank}_t{base_temp}_b{base_batch}"
            if r == base_rank
            else f"w6_rank{r}_t{base_temp}_b{base_batch}"
        )
        if existing.get(tag, {}).get("status") != "completed":
            to_rerun.append(
                {
                    "tag": tag,
                    "sweep": "rank",
                    "rank": r,
                    "temperature": base_temp,
                    "batch": base_batch,
                }
            )

    for b in [1, 2, 4, 8]:
        tag = (
            f"w6_base_r{base_rank}_t{base_temp}_b{base_batch}"
            if b == base_batch
            else f"w6_batch{b}_r{base_rank}_t{base_temp}"
        )
        if existing.get(tag, {}).get("status") != "completed":
            to_rerun.append(
                {
                    "tag": tag,
                    "sweep": "batch",
                    "rank": base_rank,
                    "temperature": base_temp,
                    "batch": b,
                }
            )

    return to_rerun


def main():
    # Peek at the results file without pulling in the heavy runner module
    # so this script exits fast when there is nothing to do.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    default_results = os.path.join(
        repo_root,
        "experiments",
        "tinker-runs",
        "results",
        "wave6_ablations.json",
    )
    results_path = os.environ.get("WAVE6_RESULTS_PATH", default_results)

    if not os.path.exists(results_path):
        print(
            f"No existing results file at {results_path}; "
            f"run wave6_ablations.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    existing = {r["tag"]: r for r in data.get("runs", [])}

    base = data.get("metadata", {}).get("baseline", {})
    base_rank = base.get("rank", 32)
    base_temp = base.get("temperature", 0.8)
    base_batch = base.get("batch", 2)

    to_rerun = _compute_rerun_plan(existing, base_rank, base_temp, base_batch)

    if not to_rerun:
        print("Nothing to rerun — every sweep entry is already completed.")
        return

    print(f"Re-running {len(to_rerun)} experiments")
    for e in to_rerun:
        print(
            f"  - {e['tag']} ({e['sweep']}) rank={e['rank']} "
            f"temp={e['temperature']} batch={e['batch']}"
        )

    # Only now do we import the heavy runner (triggers GSM8K + Tinker).
    sys.path.insert(0, here)
    from wave6_ablations import run_one  # noqa: E402

    new_results = []
    max_parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {ex.submit(run_one, e): e for e in to_rerun}
        for fut in as_completed(futures):
            try:
                new_results.append(fut.result())
            except Exception as e:
                exp = futures[fut]
                new_results.append(
                    {
                        "tag": exp["tag"],
                        "sweep": exp["sweep"],
                        "status": "crashed",
                        "error": str(e),
                    }
                )

    # Merge: only successful reruns replace existing entries.
    all_runs_by_tag = dict(existing)
    for r in new_results:
        if r.get("status") == "completed" or r["tag"] not in all_runs_by_tag:
            all_runs_by_tag[r["tag"]] = r

    merged_runs = list(all_runs_by_tag.values())

    def _collect(sweep_name, key, values):
        rows = []
        for v in values:
            if sweep_name == "temperature":
                tag = (
                    f"w6_base_r{base_rank}_t{base_temp}_b{base_batch}"
                    if v == base_temp
                    else f"w6_temp{v}_r{base_rank}_b{base_batch}"
                )
            elif sweep_name == "rank":
                tag = (
                    f"w6_base_r{base_rank}_t{base_temp}_b{base_batch}"
                    if v == base_rank
                    else f"w6_rank{v}_t{base_temp}_b{base_batch}"
                )
            else:
                tag = (
                    f"w6_base_r{base_rank}_t{base_temp}_b{base_batch}"
                    if v == base_batch
                    else f"w6_batch{v}_r{base_rank}_t{base_temp}"
                )
            r = all_runs_by_tag.get(tag)
            if r:
                rows.append(
                    {
                        key: v,
                        **{
                            k: r.get(k)
                            for k in [
                                "status",
                                "peak_reward",
                                "last10_avg",
                                "first5_avg",
                                "zero_reward_pct",
                                "zero_loss_pct",
                                "reward_trace",
                                "step_log",
                                "run_id",
                                "checkpoint",
                                "seed",
                                "rank",
                                "temperature",
                                "batch",
                                "group_size",
                                "lr",
                                "steps",
                                "wall_clock_sec",
                                "tag",
                            ]
                        },
                    }
                )
            else:
                rows.append({key: v, "status": "pending", "tag": tag})
        return rows

    completed = sum(1 for r in merged_runs if r.get("status") == "completed")
    failed = sum(
        1 for r in merged_runs if r.get("status") in ("failed", "crashed")
    )

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
            "baseline": {
                "rank": base_rank,
                "temperature": base_temp,
                "batch": base_batch,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(merged_runs),
            "completed": completed,
            "failed": failed,
            "finalized": True,
            "wandb_project": (
                "https://wandb.ai/arvindcr4-pes-university/"
                "tinker-rl-lab-world-class"
            ),
        },
        "temperature_sweep": _collect(
            "temperature", "temperature", [0.2, 0.4, 0.6, 0.8, 1.0]
        ),
        "rank_sweep": _collect("rank", "rank", [4, 8, 16, 32, 64]),
        "batch_sweep": _collect("batch", "batch", [1, 2, 4, 8]),
        "runs": merged_runs,
    }

    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone. completed={completed} failed={failed}")


if __name__ == "__main__":
    main()
