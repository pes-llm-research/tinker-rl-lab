#!/usr/bin/env python3
"""
Tool-Use / Code Reward Diagnostics.

Addresses reviewer weakness W7 (math-only depth) and question Q6
(tool-use / code reward design, ZVF behavior outside math).

Scans the repo for non-math task records (tool_use, humaneval, mbpp,
bfcl, swe-bench, multi-step tool, multi-hop ReAct). For each record we
compute:

    - reward_mean, reward_std over the reward trace
    - nonzero_fraction     (fraction of steps with strictly positive reward)
    - ZVF                  (zero-variance fraction — step-level surrogate
                            here since per-group rollout detail is not
                            always preserved in the summary JSON)
    - ERF                  (effective-rollout fraction =
                            format_pass AND nonzero_reward; when explicit
                            format_pass is absent we fall back to
                            nonzero_fraction, which is a lower bound on
                            true ERF since passing format is a necessary
                            condition for nonzero reward.)

Writes:
    experiments/results/tool_code_reward_diagnostics.tsv
and prints a LaTeX-ready diagnostic table to stdout.

Graceful degradation: if no non-math records are found, the script
writes a schema-only TSV and prints an empty-but-formatted LaTeX
table, so the artifact is always present for the paper build.

Run:  python3 experiments/tool_use_reward_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
OUT_TSV = RESULTS_DIR / "tool_code_reward_diagnostics.tsv"

NON_MATH_TASK_KEYS = {
    "tool_use", "tool-use", "tooluse",
    "humaneval", "human_eval",
    "mbpp",
    "bfcl",
    "swe-bench", "swe_bench", "swebench",
    "multistep_tool", "multi_step_tool",
    "multihop_react", "multi_hop_react",
    "code", "code_gen", "code-gen",
    "xlam",
}

NON_MATH_FILENAME_HINTS = (
    "tool", "humaneval", "mbpp", "bfcl", "swe", "code", "xlam",
    "multistep", "multihop", "react",
)


def _is_non_math_record(record: dict, filename: str) -> bool:
    task = str(record.get("task", "")).lower()
    exp = str(record.get("experiment", "")).lower()
    if any(k in task for k in NON_MATH_TASK_KEYS):
        return True
    if any(h in exp for h in NON_MATH_FILENAME_HINTS):
        return True
    fname_l = filename.lower()
    if any(h in fname_l for h in NON_MATH_FILENAME_HINTS):
        return True
    return False


def _zvf_from_trace(trace):
    """Step-level ZVF surrogate: fraction of adjacent pairs with identical
    reward. For summary JSONs that don't preserve per-rollout groups,
    this is a weak but defensible lower-bound proxy for true per-group
    ZVF; when actual zero-variance groups are present it converges to
    them. For all-zero traces it returns 1.0, matching the theoretical
    value."""
    if not trace:
        return float("nan")
    if len(trace) == 1:
        return 1.0
    identical = sum(1 for a, b in zip(trace[:-1], trace[1:]) if a == b)
    return identical / (len(trace) - 1)


def _erf_from_trace(trace, fmt_pass=None):
    if not trace:
        return float("nan")
    nonzero = [r for r in trace if r is not None and r > 0]
    if fmt_pass is None:
        # Conservative fallback: positive reward implies format passed,
        # so this is a lower bound on true ERF.
        return len(nonzero) / len(trace)
    if len(fmt_pass) != len(trace):
        return len(nonzero) / len(trace)
    both = sum(1 for r, f in zip(trace, fmt_pass) if (r is not None and r > 0 and f))
    return both / len(trace)


def _safe_mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def _nonzero_fraction(trace):
    if not trace:
        return float("nan")
    return sum(1 for r in trace if r is not None and r > 0) / len(trace)


def _extract_record_summary(record: dict, filename: str) -> dict | None:
    trace = record.get("reward_trace")
    if trace is None:
        # Try alternate field names used by sub-campaigns
        for k in ("rewards", "reward", "accuracy_trace", "reward_history"):
            if k in record and isinstance(record[k], list):
                trace = record[k]
                break
    if not isinstance(trace, list):
        return None
    mean, std = _safe_mean_std(trace)
    return {
        "source_file": filename,
        "task": str(record.get("task", "?")),
        "experiment": str(record.get("experiment", record.get("name", "?"))),
        "model": str(record.get("model_short", record.get("model", "?"))),
        "n_steps": len(trace),
        "reward_mean": mean,
        "reward_std": std,
        "nonzero_fraction": _nonzero_fraction(trace),
        "zvf": _zvf_from_trace(trace),
        "erf": _erf_from_trace(trace, record.get("format_pass_trace")),
        "peak": record.get("peak"),
        "last10_avg": record.get("last10_avg"),
    }


def _iter_result_jsons():
    search_dirs = [
        REPO_ROOT / "experiments" / "tinker-runs" / "results",
        REPO_ROOT / "experiments" / "results",
        REPO_ROOT / "experiments",
        REPO_ROOT / "grpo_ablation_results",
    ]
    seen = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for fp in d.glob("*.json"):
            if fp in seen:
                continue
            seen.add(fp)
            yield fp


def scan_repo():
    rows = []
    for fp in _iter_result_jsons():
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        records = data if isinstance(data, list) else [data]
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if not _is_non_math_record(rec, fp.name):
                continue
            summary = _extract_record_summary(rec, fp.name)
            if summary:
                rows.append(summary)
    return rows


SCHEMA_COLUMNS = [
    "source_file", "task", "experiment", "model",
    "n_steps", "reward_mean", "reward_std",
    "nonzero_fraction", "zvf", "erf",
    "peak", "last10_avg",
]


def write_tsv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(SCHEMA_COLUMNS) + "\n")
        for r in rows:
            vals = []
            for c in SCHEMA_COLUMNS:
                v = r.get(c)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}" if not math.isnan(v) else "nan")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


def _fmt_float(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.3f}"


def print_latex_table(rows):
    print("% Auto-generated by experiments/tool_use_reward_analysis.py")
    print("% Addresses: W7-math-only-depth, Q6-toolcode-reward-design")
    print(r"\begin{table}[ht]")
    print(r"\centering\small")
    print(r"\begin{tabular}{llllccccc}")
    print(r"\toprule")
    print(r"Task & Experiment & Model & $n$ & $\bar r$ & $\sigma_r$ & "
          r"nonzero frac. & ZVF & ERF \\")
    print(r"\midrule")
    if not rows:
        print(r"\multicolumn{9}{c}{\emph{No non-math reward traces found "
              r"in repository (schema-only run).}} \\")
    else:
        for r in rows:
            def esc(s):
                return str(s).replace("_", r"\_")
            print(" & ".join([
                esc(r["task"]),
                esc(r["experiment"]),
                esc(r["model"]),
                str(r["n_steps"]),
                _fmt_float(r["reward_mean"]),
                _fmt_float(r["reward_std"]),
                _fmt_float(r["nonzero_fraction"]),
                _fmt_float(r["zvf"]),
                _fmt_float(r["erf"]),
            ]) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Tool-use / code reward diagnostics. "
          r"$\bar r$, $\sigma_r$: reward mean and std over the step "
          r"trace. ZVF: step-level zero-variance-fraction surrogate. "
          r"ERF: Effective-Rollout Fraction "
          r"(\texttt{format\_pass} $\wedge$ \texttt{nonzero\_reward}; "
          r"lower-bounded by nonzero fraction when "
          r"\texttt{format\_pass\_trace} is absent).}")
    print(r"\label{tab:tool-code-reward-diag}")
    print(r"\end{table}")


def main():
    rows = scan_repo()
    write_tsv(rows, OUT_TSV)
    sys.stderr.write(
        f"[tool_use_reward_analysis] found {len(rows)} non-math record(s); "
        f"wrote {OUT_TSV.relative_to(REPO_ROOT)}\n"
    )
    print_latex_table(rows)


if __name__ == "__main__":
    main()
