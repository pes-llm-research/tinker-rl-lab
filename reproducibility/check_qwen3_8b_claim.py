#!/usr/bin/env python3
"""
Qwen3-8B headline-claim reproducibility check.

The NeurIPS submission claims (Table in paper/main.tex line 906-907, abstract F4):
    GRPO on Qwen3-8B, last-10 = 34.4%
    PPO  on Qwen3-8B, last-10 = 22.5%

A live Qwen3-8B retrain costs ~5 GPU-hours on an H100. The reviewer smoke test
cannot repeat that. Instead we recompute the last-10 mean from the **raw
per-step reward trace** stored in `experiments/master_results.json` and require
the recomputed value to agree with the published claim to within ±2
percentage points (the "±2%" spec in Task 13 step 5).

Exit 0 iff both PPO and GRPO deltas are below the 2-pp threshold.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "experiments" / "master_results.json"

# Published claim (see paper/main.tex L906-907, abstract L26-27, intro L69)
CLAIM = {
    "grpo": 0.344,  # 34.4% last-10 on Qwen3-8B / GSM8K, Tinker, G=8, seed 42
    "ppo": 0.225,  # 22.5% last-10 on Qwen3-8B / GSM8K, Modal H100
}
# Which master_results rows correspond
ROW_NAMES = {
    "grpo": "scale_gsm8k_qwen3-8b",
    "ppo": "ppo_qwen3-8b",
}
TOLERANCE_PP = 2.0  # percentage points


def _row_key(r):
    for k in ("name", "experiment_id", "tag", "experiment"):
        if k in r and r[k]:
            return r[k]
    return None


def load_rows():
    d = json.loads(MASTER.read_text())
    index = {}
    for r in d["experiments"]:
        key = _row_key(r)
        if key is None:
            continue
        # Prefer rows that carry a reward_trace when duplicates collide
        if key in index and not r.get("reward_trace"):
            continue
        index[key] = r
    return index


def recompute_last10(row) -> float:
    trace = row.get("reward_trace") or []
    if len(trace) < 10:
        raise RuntimeError(
            f"row {_row_key(row)} has only {len(trace)} steps — cannot compute last-10"
        )
    return mean(trace[-10:])


def main() -> int:
    rows = load_rows()
    report = []
    all_ok = True
    for algo, name in ROW_NAMES.items():
        row = rows.get(name)
        if row is None:
            report.append({"algo": algo, "error": f"row {name} missing"})
            all_ok = False
            continue
        recomputed = recompute_last10(row)
        claimed = CLAIM[algo]
        delta_pp = abs(recomputed - claimed) * 100.0
        ok = delta_pp <= TOLERANCE_PP
        if not ok:
            all_ok = False
        report.append(
            {
                "algo": algo,
                "row": name,
                "claimed_last10": claimed,
                "recomputed_last10": round(recomputed, 4),
                "delta_percentage_points": round(delta_pp, 3),
                "tolerance_pp": TOLERANCE_PP,
                "passed": ok,
            }
        )

    summary = {
        "claim": "Qwen3-8B / GSM8K — GRPO last-10 = 34.4%, PPO last-10 = 22.5% (Task 13 step 5 headline claim)",
        "tolerance_pp": TOLERANCE_PP,
        "all_passed": all_ok,
        "checks": report,
    }
    out = ROOT / "reproducibility" / "qwen3_8b_claim_check.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
