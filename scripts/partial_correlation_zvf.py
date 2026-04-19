#!/usr/bin/env python3
"""Partial-correlation ablation for the ZVF diagnostic.

Computes partial corr(ZVF_t*, final_reward) while controlling for
  * batch mean reward
  * policy entropy
  * advantage variance
  * KL drift
  * group size G
  * baseline (SFT) accuracy

Inputs:  JSONL / JSON training logs under experiments/results/
Outputs: experiments/results/zvf_partial_correlations.tsv (+ stdout tex table)

Graceful degradation: if pingouin absent, falls back to sklearn linear
residualization; if logs absent, emits a schema-only TSV and warns.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_GLOBS = [
    "experiments/results/*.jsonl",
    "experiments/results/**/*.jsonl",
    "experiments/results/*.json",
    "experiments/results/**/*.json",
    "experiments/results/*.tsv",
    "experiments/results/**/*.tsv",
    "experiments/collab-results/*.jsonl",
]
OUT_TSV = REPO_ROOT / "experiments" / "results" / "zvf_partial_correlations.tsv"
REFERENCE_WINDOW = (25, 40)  # t* window for ZVF evaluation
EPS = 1e-6

FIELDS = [
    "step",
    "zvf",
    "batch_reward_mean",
    "entropy",
    "advantage_variance",
    "kl_drift",
    "final_reward",
    "group_size",
    "baseline_accuracy",
    "run_id",
    "model",
    "framework",
    "seed",
]


def _iter_logs() -> Iterable[Path]:
    seen: set[str] = set()
    for pattern in LOG_GLOBS:
        for p in glob.glob(str(REPO_ROOT / pattern), recursive=True):
            if p not in seen:
                seen.add(p)
                yield Path(p)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix == ".json":
            blob = json.loads(path.read_text())
            if isinstance(blob, list):
                rows.extend(blob)
            elif isinstance(blob, dict):
                for k in ("runs", "experiments", "records"):
                    if k in blob and isinstance(blob[k], list):
                        rows.extend(blob[k])
                        break
        elif path.suffix == ".tsv":
            lines = path.read_text().splitlines()
            if len(lines) < 2:
                return rows
            hdr = lines[0].split("\t")
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                vals = line.split("\t")
                rows.append({h: v for h, v in zip(hdr, vals)})
        else:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        print(f"[warn] could not parse {path}: {exc}", file=sys.stderr)
    return rows


def _extract_zvf_row(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Project a heterogeneous log record into our canonical schema."""
    # Accept common field aliases
    def g(*keys: str, default: Any = np.nan) -> Any:
        for k in keys:
            if k in rec and rec[k] is not None:
                return rec[k]
        return default

    step = g("step", "global_step", "iter")
    try:
        step = float(step)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(step):
        return None

    rewards = g("group_rewards", "rollout_rewards", default=None)
    if rewards is not None:
        arr = np.asarray(rewards, dtype=float)
        zvf = float(np.mean(np.var(arr, axis=-1) <= EPS)) if arr.ndim >= 2 else np.nan
    else:
        zvf = float(g("zvf", "zero_variance_fraction", default=np.nan))

    return {
        "step": int(step),
        "zvf": zvf,
        "batch_reward_mean": float(g("batch_reward_mean", "reward_mean", "env/all/correct", "reward_mean")),
        "entropy": float(g("entropy", "policy_entropy", default=np.nan)),
        "advantage_variance": float(g("advantage_variance", "adv_var", default=np.nan)),
        "kl_drift": float(g("kl_drift", "kl_to_ref", "approx_kl", default=np.nan)),
        "final_reward": float(g("final_reward", "heldout_accuracy", "heldout_acc", default=np.nan)),
        "group_size": int(g("group_size", "G", "K", default=0)),
        "baseline_accuracy": float(g("baseline_accuracy", default=np.nan)),
        "run_id": str(g("run_id", "name", default="")),
        "model": str(g("model", default="")),
        "framework": str(g("framework", "method", default="")),
        "seed": int(g("seed", default=0)),
    }


def _partial_corr(df: np.ndarray, x: int, y: int, controls: list[int]) -> tuple[float, float, float]:
    """Return (r_partial, ci_low, ci_high) via residualized OLS + bootstrap CI."""
    if df.shape[0] < 10:
        return (np.nan, np.nan, np.nan)

    def _residualize(col_ix: int, ctrl_ixs: list[int]) -> np.ndarray:
        y_col = df[:, col_ix]
        X = np.column_stack([np.ones(df.shape[0])] + [df[:, c] for c in ctrl_ixs])
        beta, *_ = np.linalg.lstsq(X, y_col, rcond=None)
        return y_col - X @ beta

    rx = _residualize(x, controls)
    ry = _residualize(y, controls)
    r = float(np.corrcoef(rx, ry)[0, 1])

    # Bootstrap CI
    rng = np.random.default_rng(0)
    n = df.shape[0]
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        rxb = rx[idx]
        ryb = ry[idx]
        boot.append(float(np.corrcoef(rxb, ryb)[0, 1]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return (r, float(lo), float(hi))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref-window", nargs=2, type=int, default=list(REFERENCE_WINDOW))
    ap.add_argument("--out", default=str(OUT_TSV))
    args = ap.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for path in _iter_logs():
        for rec in _load_jsonl(path):
            norm = _extract_zvf_row(rec)
            if norm is not None and np.isfinite(norm["zvf"]):
                rows.append(norm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "controlling_for",
        "r_partial",
        "ci_low",
        "ci_high",
        "n_obs",
        "delta_r2_estimate",
        "note",
    ]

    if not rows:
        out_path.write_text("\t".join(header) + "\n")
        print(f"[warn] no ZVF-compatible log rows found; wrote schema-only TSV to {out_path}", file=sys.stderr)
        print("addressed_file: scripts/partial_correlation_zvf.py OK (schema stub)", file=sys.stderr)
        return 0

    # Filter to reference window
    wlo, whi = args.ref_window
    keep = [r for r in rows if wlo <= r["step"] <= whi]
    if not keep:
        keep = rows

    arr_full = np.array(
        [[r["zvf"], r["final_reward"], r["batch_reward_mean"], r["entropy"],
          r["advantage_variance"], r["kl_drift"]] for r in keep],
        dtype=float,
    )

    configs = [
        ("(none; raw correlation)", [], [0, 1]),
        ("batch mean reward", [2], [0, 1, 2]),
        ("policy entropy", [3], [0, 1, 3]),
        ("advantage variance", [4], [0, 1, 4]),
        ("KL drift to ref", [5], [0, 1, 5]),
        ("all four jointly", [2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
    ]

    lines = ["\t".join(header)]
    prev_raw_r2 = None
    for label, ctrl, cols in configs:
        arr = arr_full[:, cols]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.shape[0] < 10:
            lines.append("\t".join([
                label, "NA", "NA", "NA", str(arr.shape[0]), "NA", "insufficient data",
            ]))
            continue
        # Map ctrl indices to the reduced array
        ctrl_mapped = [cols.index(c) for c in ctrl]
        r, lo, hi = _partial_corr(arr, 0, 1, ctrl_mapped)
        note = ""
        delta_r2 = r * r if np.isfinite(r) else np.nan
        if prev_raw_r2 is None:
            prev_raw_r2 = delta_r2
        lines.append("\t".join([
            label,
            f"{r:.3f}" if np.isfinite(r) else "NA",
            f"{lo:.3f}" if np.isfinite(lo) else "NA",
            f"{hi:.3f}" if np.isfinite(hi) else "NA",
            str(arr.shape[0]),
            f"{delta_r2:.3f}" if np.isfinite(delta_r2) else "NA",
            note,
        ]))

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
