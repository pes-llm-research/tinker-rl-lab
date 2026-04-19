#!/usr/bin/env python3
"""Cross-framework ZVF (Zero-Variance Fraction) computation pipeline.

Implements the reference pseudocode from
``paper/sections/zvf_pipeline_spec.tex`` and the per-framework field
mappings of Table ``tab:zvf-parser-fields``:

    Framework  Reward key             Group boundary                 Mask
    -----------------------------------------------------------------------
    TRL        rewards (list/batch)   batch_size / group_size        completion_mask
    TINKER     rollout.reward         rollout.group_id               rollout.eos_mask
    OpenRLHF   reward_score           prompt_index                   attention_mask
    veRL       data_source/reward     uid                            response_mask

Each parser lifts a training log (JSON or JSONL) into a uniform
``[step, groups, K]`` reward tensor, then applies the canonical ZVF
formula

    ZVF_t = mean_g 1[ Var_K(r_{g,:}) <= eps ]

Emits a time-series JSONL (one line per optimizer step) of
``{step, zvf, batch_reward_mean, n_groups, K, framework}``.

Usage
-----

    python3 scripts/zvf_compute_cross_framework.py \
        --framework {trl,tinker,openrlhf,verl} \
        --log-path PATH [--epsilon 1e-6] [--out PATH]

    python3 scripts/zvf_compute_cross_framework.py --self-test

Graceful degradation: unknown/missing keys print a schema-expected-vs-found
diff on stderr and the step is skipped (not fatal).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy always present in repo
    np = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_FRAMEWORKS = ("trl", "tinker", "openrlhf", "verl")
DEFAULT_EPS = 1e-6

OUTPUT_FIELDS = ("step", "zvf", "batch_reward_mean", "n_groups", "K", "framework")


# ---------------------------------------------------------------------------
# Canonical ZVF primitive
# ---------------------------------------------------------------------------

def zvf(rewards_2d, eps: float = DEFAULT_EPS) -> float:
    """Canonical pseudocode from zvf_pipeline_spec.tex.

    ``rewards_2d`` is an iterable of iterables, shape ``[num_groups, K]``.
    Returns the fraction of groups whose sample variance is <= eps.
    Falls back to pure-python if numpy is unavailable.
    """
    if np is not None:
        arr = np.asarray(rewards_2d, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            # degenerate: K<=1 cannot define within-group variance
            if arr.size == 0:
                return 0.0
            return 1.0  # all groups collapse trivially
        var_per_group = arr.var(axis=-1, ddof=1)
        return float((var_per_group <= eps).mean())
    # python fallback
    groups = list(rewards_2d)
    if not groups:
        return 0.0
    flagged = 0
    for row in groups:
        row = list(row)
        k = len(row)
        if k < 2:
            flagged += 1
            continue
        mean = sum(row) / k
        var = sum((v - mean) ** 2 for v in row) / (k - 1)
        if var <= eps:
            flagged += 1
    return flagged / len(groups)


def batch_reward_mean(rewards_2d) -> float:
    if np is not None:
        arr = np.asarray(rewards_2d, dtype=float)
        return float(arr.mean()) if arr.size else 0.0
    flat: List[float] = []
    for row in rewards_2d:
        flat.extend(list(row))
    return sum(flat) / len(flat) if flat else 0.0


# ---------------------------------------------------------------------------
# Log readers
# ---------------------------------------------------------------------------

def _iter_json_records(log_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield dict records from a JSON or JSONL file.

    Handles three shapes gracefully:
      * JSONL (one json object per line)
      * JSON with a top-level list
      * JSON with a top-level dict carrying a ``steps`` / ``records`` list
    """
    try:
        text = log_path.read_text()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"log path not found: {log_path}") from e
    stripped = text.strip()
    if not stripped:
        return
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
            for item in arr:
                if isinstance(item, dict):
                    yield item
            return
        except Exception:
            pass
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ("steps", "records", "log", "training_log"):
                if isinstance(obj.get(key), list):
                    for item in obj[key]:
                        if isinstance(item, dict):
                            yield item
                    return
            # fall back: yield the dict itself if it looks step-like
            if any(k in obj for k in ("step", "global_step", "iter")):
                yield obj
                return
    # JSONL path
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(
                f"[zvf] WARN: {log_path}:{lineno} skipping malformed JSON: {e}",
                file=sys.stderr,
            )
            continue
        if isinstance(rec, dict):
            yield rec


# ---------------------------------------------------------------------------
# Schema discovery helper
# ---------------------------------------------------------------------------

def _schema_diff(expected: Iterable[str], record: Dict[str, Any]) -> str:
    expected_set = set(expected)
    found = set(_flatten_keys(record))
    missing = sorted(expected_set - found)
    extra = sorted(found - expected_set)[:8]
    return (
        f"expected={sorted(expected_set)}"
        f" missing={missing}"
        f" sample_found={extra}"
    )


def _flatten_keys(obj: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            yield new_prefix
            yield from _flatten_keys(v, new_prefix)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        yield from _flatten_keys(obj[0], prefix)


def _get_nested(obj: Any, dotted: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _as_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for v in x:
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, dict) and "reward" in v:
                try:
                    out.append(float(v["reward"]))
                except Exception:
                    continue
        return out
    return []


# ---------------------------------------------------------------------------
# Per-framework parsers
# ---------------------------------------------------------------------------

def parse_trl(record: Dict[str, Any]) -> Optional[Tuple[int, List[List[float]]]]:
    """TRL GRPOTrainer log row.

    Expects ``rewards`` (flat list), ``batch_size`` and ``group_size`` OR
    a ``group_size``/``num_generations`` hint.
    """
    step = record.get("step", record.get("global_step", record.get("iter")))
    if step is None:
        return None
    rewards = record.get("rewards")
    if rewards is None:
        rewards = record.get("rewards/mean_rewards")  # HF logger variant
    rewards = _as_float_list(rewards)
    if not rewards:
        print(
            f"[zvf][trl] schema miss @ step={step}: {_schema_diff(['rewards','group_size'], record)}",
            file=sys.stderr,
        )
        return None
    group_size = (
        record.get("group_size")
        or record.get("num_generations")
        or record.get("num_rollouts")
    )
    batch_size = record.get("batch_size") or record.get("prompts_per_batch")
    if group_size is None and batch_size is None:
        # infer square root fallback
        group_size = 1
    if group_size is None and batch_size:
        group_size = max(1, len(rewards) // int(batch_size))
    try:
        group_size = int(group_size) if group_size else 1
    except Exception:
        group_size = 1
    if group_size <= 0 or len(rewards) % group_size != 0:
        # still best-effort: trim to the nearest multiple of G
        trim = (len(rewards) // max(group_size, 1)) * max(group_size, 1)
        rewards = rewards[:trim]
        if not rewards:
            return None
    groups = [
        rewards[i : i + group_size] for i in range(0, len(rewards), group_size)
    ]
    return int(step), groups


def parse_tinker(record: Dict[str, Any]) -> Optional[Tuple[int, List[List[float]]]]:
    """TINKER managed-runtime log row.

    Expects ``rollouts`` or ``rollout`` list; each entry carries
    ``reward`` and ``group_id``.
    """
    step = record.get("step", record.get("global_step", record.get("iter")))
    if step is None:
        return None
    rollouts = record.get("rollouts")
    if rollouts is None:
        rollouts = record.get("rollout")
    if rollouts is None and "group_rewards" in record:
        # alternate shape already grouped
        gr = record["group_rewards"]
        if isinstance(gr, list) and gr and isinstance(gr[0], list):
            return int(step), [list(map(float, row)) for row in gr]
    if not isinstance(rollouts, list) or not rollouts:
        print(
            f"[zvf][tinker] schema miss @ step={step}: {_schema_diff(['rollouts.reward','rollouts.group_id'], record)}",
            file=sys.stderr,
        )
        return None
    by_group: Dict[Any, List[float]] = {}
    for idx, r in enumerate(rollouts):
        if not isinstance(r, dict):
            continue
        gid = r.get("group_id", r.get("prompt_id", idx))
        rv = r.get("reward")
        if rv is None:
            continue
        try:
            by_group.setdefault(gid, []).append(float(rv))
        except Exception:
            continue
    if not by_group:
        return None
    # preserve deterministic order by first-seen group id
    groups = [by_group[k] for k in list(by_group.keys())]
    return int(step), groups


def parse_openrlhf(record: Dict[str, Any]) -> Optional[Tuple[int, List[List[float]]]]:
    """OpenRLHF log row.

    Expects per-sample ``reward_score`` and ``prompt_index`` (or
    ``prompt_id``) either as parallel lists or an array of dicts.
    """
    step = record.get("step", record.get("global_step", record.get("iter")))
    if step is None:
        return None
    # shape 1: parallel lists
    scores = record.get("reward_score")
    indices = record.get("prompt_index", record.get("prompt_id"))
    if isinstance(scores, list) and isinstance(indices, list) and scores and indices:
        if len(scores) == len(indices):
            by_group: Dict[Any, List[float]] = {}
            for pid, s in zip(indices, scores):
                try:
                    by_group.setdefault(pid, []).append(float(s))
                except Exception:
                    continue
            if by_group:
                groups = [by_group[k] for k in list(by_group.keys())]
                return int(step), groups
    # shape 2: list of dicts
    samples = record.get("samples") or record.get("batch")
    if isinstance(samples, list) and samples:
        by_group = {}
        for s in samples:
            if not isinstance(s, dict):
                continue
            pid = s.get("prompt_index", s.get("prompt_id"))
            rv = s.get("reward_score", s.get("reward"))
            if pid is None or rv is None:
                continue
            try:
                by_group.setdefault(pid, []).append(float(rv))
            except Exception:
                continue
        if by_group:
            groups = [by_group[k] for k in list(by_group.keys())]
            return int(step), groups
    print(
        f"[zvf][openrlhf] schema miss @ step={step}: {_schema_diff(['reward_score','prompt_index'], record)}",
        file=sys.stderr,
    )
    return None


def parse_verl(record: Dict[str, Any]) -> Optional[Tuple[int, List[List[float]]]]:
    """veRL log row.

    Expects per-sample entries carrying ``uid`` (group boundary) and a
    ``data_source/reward`` or ``reward`` field. Accepts either a batch
    list under ``batch``/``samples`` or parallel lists ``uid`` and
    ``data_source/reward``.
    """
    step = record.get("step", record.get("global_step", record.get("iter")))
    if step is None:
        return None
    uids = record.get("uid")
    rewards = record.get("data_source/reward")
    if rewards is None:
        rewards = _get_nested(record, "data_source.reward")
    if isinstance(uids, list) and isinstance(rewards, list) and uids and rewards:
        if len(uids) == len(rewards):
            by_group: Dict[Any, List[float]] = {}
            for u, r in zip(uids, rewards):
                try:
                    by_group.setdefault(u, []).append(float(r))
                except Exception:
                    continue
            if by_group:
                groups = [by_group[k] for k in list(by_group.keys())]
                return int(step), groups
    # list-of-dicts shape
    batch = record.get("batch") or record.get("samples") or record.get("rollouts")
    if isinstance(batch, list) and batch:
        by_group = {}
        for s in batch:
            if not isinstance(s, dict):
                continue
            uid = s.get("uid")
            rv = s.get("data_source/reward")
            if rv is None:
                rv = _get_nested(s, "data_source.reward")
            if rv is None:
                rv = s.get("reward")
            if uid is None or rv is None:
                continue
            try:
                by_group.setdefault(uid, []).append(float(rv))
            except Exception:
                continue
        if by_group:
            groups = [by_group[k] for k in list(by_group.keys())]
            return int(step), groups
    print(
        f"[zvf][verl] schema miss @ step={step}: {_schema_diff(['uid','data_source/reward'], record)}",
        file=sys.stderr,
    )
    return None


PARSERS = {
    "trl": parse_trl,
    "tinker": parse_tinker,
    "openrlhf": parse_openrlhf,
    "verl": parse_verl,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def compute_time_series(
    records: Iterable[Dict[str, Any]],
    framework: str,
    epsilon: float = DEFAULT_EPS,
) -> List[Dict[str, Any]]:
    if framework not in PARSERS:
        raise ValueError(
            f"unknown framework {framework!r}; choose from {SUPPORTED_FRAMEWORKS}"
        )
    parser = PARSERS[framework]
    rows: List[Dict[str, Any]] = []
    for rec in records:
        parsed = parser(rec)
        if parsed is None:
            continue
        step, groups = parsed
        if not groups:
            continue
        # normalise to padded 2D if K varies: use minimum K across groups
        k_min = min(len(g) for g in groups)
        if k_min <= 0:
            continue
        trimmed = [g[:k_min] for g in groups if len(g) >= k_min]
        if not trimmed:
            continue
        rows.append(
            {
                "step": int(step),
                "zvf": zvf(trimmed, eps=epsilon),
                "batch_reward_mean": batch_reward_mean(trimmed),
                "n_groups": len(trimmed),
                "K": k_min,
                "framework": framework,
            }
        )
    rows.sort(key=lambda r: r["step"])
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def default_out_path(log_path: Path, framework: str) -> Path:
    stem = log_path.stem
    return REPO_ROOT / "experiments" / "results" / f"zvf_{framework}_{stem}.jsonl"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    errors: List[str] = []

    # TRL fixture: 3 groups of 2 rollouts => 6 rewards flat
    trl_log = [
        {"step": 0, "group_size": 2, "batch_size": 3, "rewards": [0.0, 1.0, 0.5, 0.5, 1.0, 1.0]},
        {"step": 1, "group_size": 2, "batch_size": 3, "rewards": [1.0, 1.0, 0.0, 0.0, 0.5, 0.75]},
    ]
    rows = compute_time_series(trl_log, "trl")
    # group (1,1) and (0,0) both zero-variance, (0.5,0.5) zero-variance too
    # step 0: groups [0,1] var 0.5, [0.5,0.5] var 0, [1,1] var 0 -> ZVF=2/3
    # step 1: [1,1] var 0, [0,0] var 0, [0.5,0.75] var>0 -> ZVF=2/3
    if len(rows) != 2:
        errors.append(f"trl rows={len(rows)} != 2")
    else:
        if abs(rows[0]["zvf"] - 2 / 3) > 1e-9:
            errors.append(f"trl step0 zvf={rows[0]['zvf']} expected {2/3}")
        if rows[0]["n_groups"] != 3 or rows[0]["K"] != 2:
            errors.append(f"trl shape wrong: {rows[0]}")

    # TINKER fixture
    tinker_log = [
        {
            "step": 0,
            "rollouts": [
                {"group_id": "A", "reward": 1.0},
                {"group_id": "A", "reward": 1.0},
                {"group_id": "B", "reward": 0.0},
                {"group_id": "B", "reward": 1.0},
                {"group_id": "C", "reward": 0.0},
                {"group_id": "C", "reward": 0.0},
            ],
        }
    ]
    rows = compute_time_series(tinker_log, "tinker")
    if len(rows) != 1 or abs(rows[0]["zvf"] - 2 / 3) > 1e-9:
        errors.append(f"tinker unexpected rows={rows}")
    if rows and rows[0]["framework"] != "tinker":
        errors.append("tinker framework tag missing")

    # OpenRLHF fixture: parallel lists
    openrlhf_log = [
        {
            "step": 7,
            "prompt_index": [0, 0, 1, 1, 2, 2, 3, 3],
            "reward_score": [1.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.25, 0.75],
        }
    ]
    rows = compute_time_series(openrlhf_log, "openrlhf")
    # groups: (1,1) 0-var, (0,1) >0, (0.5,0.5) 0-var, (0.25,0.75) >0 -> 2/4
    if len(rows) != 1 or abs(rows[0]["zvf"] - 0.5) > 1e-9:
        errors.append(f"openrlhf unexpected: {rows}")

    # veRL fixture: list-of-dicts batch with data_source/reward
    verl_log = [
        {
            "step": 42,
            "batch": [
                {"uid": "u0", "data_source/reward": 1.0},
                {"uid": "u0", "data_source/reward": 1.0},
                {"uid": "u1", "data_source/reward": 0.25},
                {"uid": "u1", "data_source/reward": 0.75},
            ],
        }
    ]
    rows = compute_time_series(verl_log, "verl")
    if len(rows) != 1 or abs(rows[0]["zvf"] - 0.5) > 1e-9:
        errors.append(f"verl unexpected: {rows}")

    # Epsilon sensitivity
    r = zvf([[0.0, 1e-9], [0.5, 0.5]], eps=1e-6)
    if abs(r - 1.0) > 1e-9:
        errors.append(f"epsilon sensitivity wrong: {r}")
    r_tight = zvf([[0.0, 1e-9], [0.5, 0.5]], eps=1e-20)
    if abs(r_tight - 0.5) > 1e-9:
        errors.append(f"epsilon tight wrong: {r_tight}")

    # Round-trip write/read
    with tempfile.TemporaryDirectory() as td:
        fixture = Path(td) / "trl_fixture.jsonl"
        with fixture.open("w") as f:
            for rec in trl_log:
                f.write(json.dumps(rec) + "\n")
        out = Path(td) / "out.jsonl"
        rs = compute_time_series(_iter_json_records(fixture), "trl")
        write_jsonl(rs, out)
        got = [json.loads(l) for l in out.read_text().splitlines()]
        if len(got) != 2 or set(got[0].keys()) != set(OUTPUT_FIELDS):
            errors.append(f"round-trip wrong: {got}")

    if errors:
        print("SELF-TEST FAILED:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1
    print(
        "self-test OK: parsers {trl,tinker,openrlhf,verl} + eps sensitivity + round-trip",
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compute cross-framework ZVF time-series from training logs.",
    )
    p.add_argument("--framework", choices=SUPPORTED_FRAMEWORKS)
    p.add_argument("--log-path", type=Path, help="Path to the framework's training log")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPS)
    p.add_argument("--out", type=Path, default=None, help="Output JSONL path")
    p.add_argument("--self-test", action="store_true", help="Run internal fixtures")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.framework or not args.log_path:
        p.error("--framework and --log-path are required (or pass --self-test)")
        return 2

    log_path: Path = args.log_path
    if not log_path.is_absolute():
        log_path = (REPO_ROOT / log_path).resolve()
    if not log_path.exists():
        print(f"[zvf] ERROR: log path does not exist: {log_path}", file=sys.stderr)
        return 3

    out_path: Path = args.out or default_out_path(log_path, args.framework)
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    records = _iter_json_records(log_path)
    rows = compute_time_series(records, args.framework, epsilon=args.epsilon)
    write_jsonl(rows, out_path)
    print(
        f"[zvf] {args.framework}: wrote {len(rows)} step(s) to {out_path}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
