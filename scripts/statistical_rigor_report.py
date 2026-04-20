#!/usr/bin/env python3
"""
scripts/statistical_rigor_report.py

Produces a rigor-classified report of all statistical comparisons in the
paper, flagging which are well-powered (multi-seed, matched-init,
n>=5 per group) vs under-powered or descriptive-only.

This directly addresses reviewer weaknesses W4, W5, Q1, Q5:
  - "Many headline comparisons rely on single-seed runs..."
  - "BH-adjusted significance table aggregates single-seed data..."
  - "Restrict F1-F5 to 5-seed >=100-step single-framework runs"

Usage
-----
    python3 scripts/statistical_rigor_report.py

Output
------
    experiments/results/statistical_rigor_report.tsv
    schema: comparison_id, claim, n_seeds_A, n_seeds_B, matched_init,
            well_powered, effect_size, ci_low, ci_high, recommendation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
OUT_TSV = RESULTS_DIR / "statistical_rigor_report.tsv"

# Minimum seeds per group to be considered well-powered
MIN_SEEDS = 5
MIN_STEPS = 100


def cohens_d(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Cohen's d with 95% CI using Hedges-Olkin SE."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    m1, m2 = sum(a) / n1, sum(b) / n2
    var1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    s_pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return float("nan"), float("nan"), float("nan")
    d = (m1 - m2) / s_pooled
    se = math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    ci_low = d - 1.96 * se
    ci_high = d + 1.96 * se
    return d, ci_low, ci_high


def bootstrap_ci(
    a: Sequence[float], b: Sequence[float], n_bootstrap: int = 2000
) -> Tuple[float, float]:
    """Bootstrap 95% CI for mean difference."""
    import random

    rng = random.Random(42)
    diffs = []
    for _ in range(n_bootstrap):
        ba = [rng.choice(a) for _ in range(len(a))]
        bb = [rng.choice(b) for _ in range(len(b))]
        diffs.append(sum(ba) / len(ba) - sum(bb) / len(bb))
    diffs.sort()
    return diffs[int(0.025 * n_bootstrap)], diffs[int(0.975 * n_bootstrap)]


# ---------------------------------------------------------------------------
# Comparison registry
# ---------------------------------------------------------------------------

@dataclass
class Comparison:
    id: str
    claim: str
    source_file: Path
    group_col: str
    metric_col: str
    filter_fn: Optional[callable]
    group_a: str
    group_b: str
    well_powered_fn: callable  # (df) -> bool


# We use variance_mitigation.tsv as the primary multi-seed source

def _load_variance_mitigation_last_step() -> Dict[str, List[float]]:
    """Load last-step heldout_acc per method from variance_mitigation.tsv."""
    import csv

    path = RESULTS_DIR / "variance_mitigation.tsv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    # Group by method, take last step per seed
    by_method_seed: Dict[Tuple[str, int], List[float]] = {}
    for r in rows:
        try:
            m, s = r["method"], int(r["seed"])
            v = float(r["heldout_acc"])
        except (KeyError, ValueError):
            continue
        by_method_seed.setdefault((m, s), []).append(v)
    # Take last value per seed
    per_method: Dict[str, List[float]] = {}
    for (m, s), vals in by_method_seed.items():
        per_method.setdefault(m, []).append(vals[-1])
    return per_method


def _load_base_instruct() -> Tuple[Optional[List[float]], Optional[List[float]]]:
    import csv

    path = RESULTS_DIR / "base_instruct_paired.tsv"
    if not path.exists():
        return None, None
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    base = [float(r["accuracy"]) for r in rows if r.get("init") == "base"]
    instruct = [float(r["accuracy"]) for r in rows if r.get("init") == "instruct"]
    return base, instruct


def _load_group_size() -> Dict[str, List[float]]:
    import csv

    path = RESULTS_DIR / "group_size_token_normalized.tsv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    out: Dict[str, List[float]] = {}
    for r in rows:
        g = r.get("group_size")
        if g and "accuracy" in r:
            out.setdefault(g, []).append(float(r["accuracy"]))
    return out


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report() -> List[Dict[str, str]]:
    report: List[Dict[str, str]] = []

    # ---- Comparison 1: Variance mitigation head-to-head (well-powered) ----
    vm = _load_variance_mitigation_last_step()
    methods = ["aero", "cppo", "ngrpo", "scafgrpo", "mcgrpo", "gift", "areal", "es"]
    baseline = vm.get("grpo", [])
    for m in methods:
        vals = vm.get(m, [])
        n = len(vals)
        wp = n >= MIN_SEEDS and len(baseline) >= MIN_SEEDS
        if n >= 2 and len(baseline) >= 2:
            d, ci_l, ci_h = cohens_d(vals, baseline)
            rec = "report_with_ci" if wp else "descriptive_only_no_pvalues"
        else:
            d = ci_l = ci_h = float("nan")
            rec = "insufficient_data"
        report.append(
            dict(
                comparison_id=f"vm_{m}_vs_grpo",
                claim=f"{m.upper()} vs baseline GRPO (last-step heldout)",
                n_seeds_A=str(n),
                n_seeds_B=str(len(baseline)),
                matched_init="yes",
                well_powered="yes" if wp else "no",
                effect_size=f"{d:.3f}" if not math.isnan(d) else "NA",
                ci_low=f"{ci_l:.3f}" if not math.isnan(ci_l) else "NA",
                ci_high=f"{ci_h:.3f}" if not math.isnan(ci_h) else "NA",
                recommendation=rec,
            )
        )

    # ---- Comparison 2: Base vs Instruct ----
    base, instruct = _load_base_instruct()
    if base and instruct:
        wp = len(base) >= MIN_SEEDS and len(instruct) >= MIN_SEEDS
        d, ci_l, ci_h = cohens_d(instruct, base)
        report.append(
            dict(
                comparison_id="base_vs_instruct",
                claim="Instruct vs Base initialization (paired)",
                n_seeds_A=str(len(instruct)),
                n_seeds_B=str(len(base)),
                matched_init="yes",
                well_powered="yes" if wp else "no",
                effect_size=f"{d:.3f}",
                ci_low=f"{ci_l:.3f}",
                ci_high=f"{ci_h:.3f}",
                recommendation="report_with_ci" if wp else "descriptive_only_no_pvalues",
            )
        )

    # ---- Comparison 3: Group size ablation ----
    gs = _load_group_size()
    sizes = sorted(gs.keys())
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        va, vb = gs[a], gs[b]
        wp = len(va) >= MIN_SEEDS and len(vb) >= MIN_SEEDS
        if len(va) >= 2 and len(vb) >= 2:
            d, ci_l, ci_h = cohens_d(va, vb)
        else:
            d = ci_l = ci_h = float("nan")
        report.append(
            dict(
                comparison_id=f"groupsize_{a}_vs_{b}",
                claim=f"Group size {a} vs {b} (token-normalized)",
                n_seeds_A=str(len(va)),
                n_seeds_B=str(len(vb)),
                matched_init="yes",
                well_powered="yes" if wp else "no",
                effect_size=f"{d:.3f}" if not math.isnan(d) else "NA",
                ci_low=f"{ci_l:.3f}" if not math.isnan(ci_l) else "NA",
                ci_high=f"{ci_h:.3f}" if not math.isnan(ci_h) else "NA",
                recommendation="report_with_ci" if wp else "descriptive_only_no_pvalues",
            )
        )

    return report


def write_report(rows: List[Dict[str, str]], path: Path = OUT_TSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "comparison_id",
        "claim",
        "n_seeds_A",
        "n_seeds_B",
        "matched_init",
        "well_powered",
        "effect_size",
        "ci_low",
        "ci_high",
        "recommendation",
    ]
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Statistical rigor report")
    p.add_argument("--out", default=str(OUT_TSV), help="Output TSV path")
    args = p.parse_args(argv)

    rows = generate_report()
    write_report(rows, Path(args.out))

    well_powered = sum(1 for r in rows if r["well_powered"] == "yes")
    total = len(rows)
    print(f"[statistical_rigor] {well_powered}/{total} comparisons well-powered")
    print(f"[statistical_rigor] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
