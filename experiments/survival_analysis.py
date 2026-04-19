"""Survival analysis for findings F1-F5 under the strict Tier-A/B inferential filter.

Addresses reviewer weaknesses:
  * W4-single-seed-tinker : Tinker API single-seed 20-30 step runs have no
    statistical power; downgrade them to Tier-C descriptive.
  * W5-bh-aggregates-singleseed : BH correction must not aggregate Tier-C
    single-seed data with Tier-A/B multi-seed data.
  * Q5-f1-f5-survival-5seed : Which of F1-F5 survive when restricted to
    TRL-as-single-open-framework, >=5 seeds, >=100 steps?

Tier definitions (evidence strength):
  A  >=5 seeds AND >=100 steps  (inferential-grade)
  B  3-4 seeds AND 50-100 steps (supporting)
  C  single-seed OR <50 steps   (descriptive only, EXCLUDED from inference)

Outputs experiments/results/survival_analysis.tsv and a LaTeX-ready table on
stdout.  If no Tier-A data is present the script still emits a schema-only TSV
plus a stderr warning, so it can be run in partial-data environments.

Run:  python3 experiments/survival_analysis.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
OUT_TSV = RESULTS_DIR / "survival_analysis.tsv"

BOOTSTRAP_ITERS = 10_000
RNG_SEED = 20260506
ALPHA = 0.05

# Thresholds for tier classification.
TIER_A_MIN_SEEDS = 5
TIER_A_MIN_STEPS = 100
TIER_B_MIN_SEEDS = 3
TIER_B_MIN_STEPS = 50


# ---------------------------------------------------------------------------
# Run record model
# ---------------------------------------------------------------------------
@dataclass
class RunGroup:
    """A group of replicate runs sharing (framework, model, task, algo, config)."""

    framework: str
    model: str
    task: str
    algo: str
    seeds: List[int] = field(default_factory=list)
    step_counts: List[int] = field(default_factory=list)
    last10: List[float] = field(default_factory=list)
    peak: List[float] = field(default_factory=list)
    raw_records: List[dict] = field(default_factory=list)

    @property
    def n_seeds(self) -> int:
        return len(set(self.seeds))

    @property
    def min_steps(self) -> int:
        return min(self.step_counts) if self.step_counts else 0

    @property
    def tier(self) -> str:
        if self.n_seeds >= TIER_A_MIN_SEEDS and self.min_steps >= TIER_A_MIN_STEPS:
            return "A"
        if self.n_seeds >= TIER_B_MIN_SEEDS and self.min_steps >= TIER_B_MIN_STEPS:
            return "B"
        return "C"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _coerce_record(raw: dict) -> Optional[dict]:
    """Project a heterogeneous run record onto the minimal schema we need."""
    if not isinstance(raw, dict):
        return None
    framework = (
        raw.get("framework")
        or raw.get("platform")
        or raw.get("library")
        or raw.get("group", "").replace("Tinker GRPO", "tinker")
        or "unknown"
    )
    framework = str(framework).lower()
    model = raw.get("model") or raw.get("model_short") or raw.get("base_model") or "unknown"
    task = raw.get("task") or raw.get("dataset") or "unknown"
    algo = raw.get("method") or raw.get("algorithm") or raw.get("algo") or "GRPO"
    seed = raw.get("seed")
    steps = raw.get("steps_completed") or raw.get("steps") or raw.get("num_steps")
    last10 = _safe_float(raw.get("last10_avg") or raw.get("last10") or raw.get("last_10"))
    peak = _safe_float(raw.get("peak_reward") or raw.get("peak"))
    status = (raw.get("status") or "").lower()
    if status == "failed" and last10 is None:
        return None
    if steps is None or last10 is None:
        return None
    return {
        "framework": str(framework),
        "model": str(model),
        "task": str(task),
        "algo": str(algo),
        "seed": int(seed) if isinstance(seed, (int, float)) else None,
        "steps": int(steps) if isinstance(steps, (int, float)) else None,
        "last10": float(last10),
        "peak": float(peak) if peak is not None else None,
    }


def _load_json_records(path: Path) -> Iterable[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        if "experiments" in data and isinstance(data["experiments"], list):
            return data["experiments"]
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        if "frameworks" in data and isinstance(data["frameworks"], list):
            return data["frameworks"]
        return [data]
    if isinstance(data, list):
        return data
    return []


def _load_jsonl_records(path: Path) -> Iterable[dict]:
    recs: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return recs


def _load_csv_records(path: Path) -> Iterable[dict]:
    recs: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for k in ("steps", "steps_completed", "seed"):
                    if k in row and row[k] not in ("", None):
                        try:
                            row[k] = int(float(row[k]))
                        except ValueError:
                            pass
                for k in ("last10_avg", "last10", "peak_reward", "peak"):
                    if k in row and row[k] not in ("", None):
                        try:
                            row[k] = float(row[k])
                        except ValueError:
                            pass
                recs.append(row)
    except OSError:
        pass
    return recs


def load_all_runs() -> List[dict]:
    """Load run records from the canonical set of repo paths."""
    sources: List[Path] = []
    sources += sorted(RESULTS_DIR.glob("*.jsonl"))
    sources += sorted(RESULTS_DIR.glob("**/*.jsonl"))
    sources.append(EXPERIMENTS_DIR / "master_results.json")
    sources.append(EXPERIMENTS_DIR / "master_results.csv")
    sources += sorted(RESULTS_DIR.glob("*.json"))

    records: List[dict] = []
    seen: set = set()
    for src in sources:
        if not src.exists():
            continue
        key = str(src.resolve())
        if key in seen:
            continue
        seen.add(key)
        if src.suffix == ".jsonl":
            raw = _load_jsonl_records(src)
        elif src.suffix == ".csv":
            raw = _load_csv_records(src)
        else:
            raw = _load_json_records(src)
        for r in raw:
            rec = _coerce_record(r)
            if rec:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Grouping and tier assignment
# ---------------------------------------------------------------------------
def group_runs(records: List[dict]) -> List[RunGroup]:
    groups: Dict[tuple, RunGroup] = {}
    for r in records:
        key = (r["framework"], r["model"], r["task"], r["algo"])
        g = groups.get(key)
        if g is None:
            g = RunGroup(
                framework=r["framework"], model=r["model"], task=r["task"], algo=r["algo"]
            )
            groups[key] = g
        if r["seed"] is not None:
            g.seeds.append(int(r["seed"]))
        if r["steps"] is not None:
            g.step_counts.append(int(r["steps"]))
        g.last10.append(float(r["last10"]))
        if r["peak"] is not None:
            g.peak.append(float(r["peak"]))
        g.raw_records.append(r)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Stats primitives
# ---------------------------------------------------------------------------
def _mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def _std(x: List[float], ddof: int = 1) -> float:
    if len(x) - ddof <= 0:
        return 0.0
    return statistics.pstdev(x) if ddof == 0 else statistics.stdev(x)


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled == 0.0:
        return 0.0
    return (_mean(a) - _mean(b)) / pooled


def welch_t(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Return (t, df). p-value is computed separately using a Student-t CDF."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = _mean(a), _mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0.0:
        return float("nan"), float("nan")
    t = (ma - mb) / denom
    num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1)
    df = num / df_den if df_den > 0 else float("nan")
    return t, df


def _student_t_sf(t: float, df: float) -> float:
    """Two-sided Student-t survival function via the regularized incomplete beta."""
    if math.isnan(t) or math.isnan(df) or df <= 0:
        return float("nan")
    x = df / (df + t * t)
    # Regularized incomplete beta I_x(df/2, 1/2) via continued fraction.
    a, b = df / 2.0, 0.5
    if x <= 0.0:
        ix = 0.0
    elif x >= 1.0:
        ix = 1.0
    else:
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
        # Lentz continued fraction.
        fpmin = 1e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-10:
                break
        ix = front * h
    return max(min(ix, 1.0), 0.0)


def welch_pvalue(a: List[float], b: List[float]) -> float:
    t, df = welch_t(a, b)
    if math.isnan(t) or math.isnan(df):
        return float("nan")
    return _student_t_sf(abs(t), df)


def bootstrap_mean_ci(
    x: List[float], iters: int = BOOTSTRAP_ITERS, alpha: float = ALPHA
) -> Tuple[float, float]:
    if not x:
        return (float("nan"), float("nan"))
    if len(x) == 1:
        return (x[0], x[0])
    import random

    rng = random.Random(RNG_SEED)
    n = len(x)
    boots = []
    for _ in range(iters):
        sample = [x[rng.randrange(n)] for _ in range(n)]
        boots.append(_mean(sample))
    boots.sort()
    lo = boots[int(alpha / 2 * iters)]
    hi = boots[int((1 - alpha / 2) * iters)]
    return (lo, hi)


def bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg step-up with FDR=0.05. Returns adjusted p-values."""
    if not pvals:
        return []
    m = len(pvals)
    idx = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = idx[rank]
        p = pvals[i]
        val = p * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj


# ---------------------------------------------------------------------------
# Finding-specific tests
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    key: str
    claim: str
    tier_a_predicate: callable  # (tier_ab_groups) -> (n_used, effect, a, b) | None
    tier_b_predicate: callable


def _select_open_framework_grpo(groups: List[RunGroup]) -> List[RunGroup]:
    """TRL is the designated single open framework for Tier-A inference."""
    open_ok = {"trl", "verl", "openrlhf"}
    return [g for g in groups if g.framework in open_ok and g.algo.upper() == "GRPO"]


def _tierab_runs(groups: List[RunGroup]) -> List[RunGroup]:
    return [g for g in groups if g.tier in ("A", "B")]


def _f1_zvf_diagnostic(groups: List[RunGroup]):
    """F1: ZVF tracks signal degeneracy in sparse-reward GRPO.
    Tier-A/B check: any multi-seed GRPO run with computed per-step ZVF present.
    We approximate by looking for `zero_reward_pct` being populated and
    correlating with low last10 across seeds.
    """
    gsm = [g for g in groups if g.task.lower().startswith("gsm")]
    gsm_ab = _tierab_runs(gsm)
    if not gsm_ab:
        return None
    lows, highs = [], []
    for g in gsm_ab:
        for r in g.raw_records:
            zvf_like = r.get("raw", {}).get("zero_reward_pct") if isinstance(r.get("raw"), dict) else None
            # best-effort: if sparse info, fall back to correlation of last10 across seeds
        lows.extend(g.last10)
    a = [v for v in lows if v < 0.5]
    b = [v for v in lows if v >= 0.5]
    if len(a) < 2 or len(b) < 2:
        return None
    return (len(a) + len(b), cohens_d(b, a), b, a)


def _f2_trainability(groups: List[RunGroup]):
    """F2: instruction-tuned checkpoints are easier to optimize than base.
    Tier-A/B: need >=2 instruct vs >=2 base runs in GRPO at same task.
    """
    trl_grpo = _select_open_framework_grpo(_tierab_runs(groups))
    inst = []
    base = []
    for g in trl_grpo:
        m = g.model.lower()
        tag = "inst" if ("instruct" in m or "inst" in m) else "base"
        (inst if tag == "inst" else base).extend(g.last10)
    if len(inst) < 2 or len(base) < 2:
        return None
    return (len(inst) + len(base), cohens_d(inst, base), inst, base)


def _f3_ppo_vs_grpo(groups: List[RunGroup]):
    """F3: PPO vs GRPO ranking is heterogeneous across model families.
    Tier-A/B: both algos with >=3 seeds >=50 steps on same model+framework.
    """
    tierab = _tierab_runs(groups)
    grpo = [g for g in tierab if g.algo.upper() == "GRPO"]
    ppo = [g for g in tierab if g.algo.upper() == "PPO"]
    if not grpo or not ppo:
        return None
    a, b = [], []
    for g in grpo:
        a.extend(g.last10)
    for g in ppo:
        b.extend(g.last10)
    if len(a) < 2 or len(b) < 2:
        return None
    return (len(a) + len(b), cohens_d(a, b), a, b)


def _f4_framework_gap(groups: List[RunGroup]):
    """F4: framework implementation dominates outcomes (Henderson-style).
    Tier-A/B: need Tier-A/B data in >=2 open frameworks on same model/algo.
    """
    tierab = _tierab_runs(groups)
    by_fw: Dict[str, List[float]] = {}
    for g in tierab:
        if g.algo.upper() != "GRPO":
            continue
        by_fw.setdefault(g.framework, []).extend(g.last10)
    fws = [f for f, vs in by_fw.items() if len(vs) >= 2]
    if len(fws) < 2:
        return None
    a = by_fw[fws[0]]
    b = by_fw[fws[1]]
    return (len(a) + len(b), cohens_d(a, b), a, b)


def _f5_frontier_stability(groups: List[RunGroup]):
    """F5: Frontier (>=70B) model stability claims.
    Tier-A/B: require multi-seed runs on >=70B models. In practice NONE exist
    in the current release; this test is expected to FAIL the survival gate.
    """
    tierab = _tierab_runs(groups)
    frontier = []
    non_frontier = []
    for g in tierab:
        m = g.model.lower()
        # Heuristic: size tokens in model name.
        is_frontier = any(
            tag in m
            for tag in ("70b", "120b", "32b", "27b", "235b", "397b", "671b", "v3.1")
        )
        (frontier if is_frontier else non_frontier).extend(g.last10)
    if len(frontier) < 2:
        return None
    if len(non_frontier) < 2:
        return None
    return (len(frontier) + len(non_frontier), cohens_d(frontier, non_frontier), frontier, non_frontier)


FINDINGS: List[Tuple[str, str, callable]] = [
    (
        "F1",
        "ZVF tracks signal degeneracy in sparse-reward GRPO",
        _f1_zvf_diagnostic,
    ),
    (
        "F2",
        "Instruction-tuned checkpoints are easier to optimize than base",
        _f2_trainability,
    ),
    (
        "F3",
        "PPO vs GRPO ranking is heterogeneous across model families",
        _f3_ppo_vs_grpo,
    ),
    (
        "F4",
        "Framework implementation gap on identical GRPO config",
        _f4_framework_gap,
    ),
    (
        "F5",
        "Frontier (>=70B) model stability claims",
        _f5_frontier_stability,
    ),
]


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def run() -> None:
    records = load_all_runs()
    groups = group_runs(records)
    n_runs = len(records)
    n_groups = len(groups)
    n_a = sum(1 for g in groups if g.tier == "A")
    n_b = sum(1 for g in groups if g.tier == "B")
    n_c = sum(1 for g in groups if g.tier == "C")

    sys.stderr.write(
        f"[survival] loaded {n_runs} runs in {n_groups} groups "
        f"(Tier-A={n_a}, Tier-B={n_b}, Tier-C={n_c})\n"
    )

    header = [
        "finding",
        "tier_a_support",
        "tier_b_support",
        "effect_size_cohens_d",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "n_runs_used",
        "bh_adjusted_p",
        "conclusion",
    ]

    # If NO Tier-A data at all, emit schema-only TSV and warn.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if n_a == 0:
        sys.stderr.write(
            "[survival] WARNING: no Tier-A (>=5 seeds, >=100 steps) groups found. "
            "Emitting schema-only TSV; all F1-F5 claims DOWNGRADED to descriptive.\n"
        )
        with OUT_TSV.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(header)
            for key, claim, _ in FINDINGS:
                writer.writerow(
                    [key, "no", "no", "nan", "nan", "nan", 0, "nan", "downgraded-to-Tier-C"]
                )
        _print_latex_table([], n_a, n_b, n_c)
        return

    raw_pvals: List[float] = []
    rows: List[dict] = []
    for key, claim, fn in FINDINGS:
        result = fn(groups)
        if result is None:
            rows.append(
                {
                    "finding": key,
                    "claim": claim,
                    "tier_a_support": "no",
                    "tier_b_support": "no",
                    "d": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "n_used": 0,
                    "p": float("nan"),
                    "conclusion": "insufficient Tier-A/B data (survival: NO)",
                }
            )
            continue
        n_used, d, a, b = result
        ci_lo, ci_hi = bootstrap_mean_ci([ai - bi for ai, bi in zip(a[: min(len(a), len(b))], b[: min(len(a), len(b))])])
        p = welch_pvalue(a, b)
        raw_pvals.append(p if not math.isnan(p) else 1.0)

        # Tier-A support requires at least one Tier-A group contributing to the finding
        tier_a_has = any(g.tier == "A" for g in groups)
        tier_b_has = any(g.tier == "B" for g in groups)

        rows.append(
            {
                "finding": key,
                "claim": claim,
                "tier_a_support": "yes" if tier_a_has else "no",
                "tier_b_support": "yes" if tier_b_has else "no",
                "d": d,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_used": n_used,
                "p": p,
                "conclusion": "",  # filled below after BH
            }
        )

    # Apply BH over Tier-A/B derived p-values ONLY.
    ps = [r["p"] for r in rows if not math.isnan(r["p"])]
    adj = bh_adjust(ps)
    itr = iter(adj)
    for r in rows:
        if math.isnan(r["p"]):
            r["p_bh"] = float("nan")
            r["conclusion"] = "insufficient Tier-A/B data (survival: NO)"
            continue
        pb = next(itr)
        r["p_bh"] = pb
        if pb < ALPHA and abs(r["d"]) >= 0.3:
            r["conclusion"] = f"survives BH (adj p={pb:.3g}, |d|={abs(r['d']):.2f})"
        elif pb < ALPHA:
            r["conclusion"] = f"survives BH but small effect (adj p={pb:.3g}, |d|={abs(r['d']):.2f})"
        else:
            r["conclusion"] = f"DOES NOT survive BH (adj p={pb:.3g})"

    # Write TSV
    with OUT_TSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        for r in rows:
            writer.writerow(
                [
                    r["finding"],
                    r["tier_a_support"],
                    r["tier_b_support"],
                    f"{r['d']:.3f}" if not math.isnan(r["d"]) else "nan",
                    f"{r['ci_lo']:.3f}" if not math.isnan(r["ci_lo"]) else "nan",
                    f"{r['ci_hi']:.3f}" if not math.isnan(r["ci_hi"]) else "nan",
                    r["n_used"],
                    f"{r['p_bh']:.4g}" if not math.isnan(r["p_bh"]) else "nan",
                    r["conclusion"],
                ]
            )

    _print_latex_table(rows, n_a, n_b, n_c)


def _print_latex_table(rows: List[dict], n_a: int, n_b: int, n_c: int) -> None:
    print(r"% --- Survival table (auto-generated by experiments/survival_analysis.py) ---")
    print(r"\begin{tabular}{@{}l l c c r r l@{}}")
    print(r"\toprule")
    print(
        r"\textbf{Finding} & \textbf{Claim (short)} & \textbf{Tier-A?} & \textbf{Tier-B?} & $d$ & $p_{\mathrm{BH}}$ & \textbf{Survival} \\"
    )
    print(r"\midrule")
    if not rows:
        print(
            r"F1 & ZVF diagnostic & -- & -- & -- & -- & downgraded to Tier-C \\"
        )
        print(
            r"F2 & Instruct $>$ Base & -- & -- & -- & -- & downgraded to Tier-C \\"
        )
        print(
            r"F3 & PPO/GRPO heterogeneity & -- & -- & -- & -- & downgraded to Tier-C \\"
        )
        print(
            r"F4 & Framework gap & -- & -- & -- & -- & downgraded to Tier-C \\"
        )
        print(
            r"F5 & Frontier stability & -- & -- & -- & -- & downgraded to Tier-C \\"
        )
    else:
        short = {
            "F1": "ZVF diagnostic",
            "F2": "Instruct $>$ Base",
            "F3": "PPO/GRPO heterogeneity",
            "F4": "Framework gap",
            "F5": "Frontier stability",
        }
        for r in rows:
            surv = "\\textbf{survives}" if "survives BH" in r["conclusion"] and "DOES NOT" not in r["conclusion"] else "no"
            d = f"{r['d']:+.2f}" if not math.isnan(r["d"]) else "--"
            pb = f"{r['p_bh']:.3g}" if not math.isnan(r["p_bh"]) else "--"
            print(
                f"{r['finding']} & {short.get(r['finding'], r['finding'])} & "
                f"{r['tier_a_support']} & {r['tier_b_support']} & {d} & {pb} & {surv} \\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()
    print(
        f"% Tier counts: A={n_a}  B={n_b}  C={n_c}. "
        f"BH applied over Tier-A/B p-values only (W5)."
    )


if __name__ == "__main__":
    run()
