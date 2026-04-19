#!/usr/bin/env python3
"""
scripts/regenerate_figures.py
=============================

Submission-quality figure regeneration for the Tinker-RL-Lab NeurIPS 2026
paper. Produces all eight figures referenced in ``paper/main.tex`` at
300 dpi in both PNG and PDF, writing to ``paper/figures/v2/``.

Design rules enforced throughout:
    * Okabe--Ito colorblind-safe palette (8 colors, with a dedicated role
      per series family).
    * Serif font family matched to NeurIPS body text, with mathtext set to
      the ``stix`` family so inline math renders in the same face.
    * Legends live OUTSIDE the axes (bbox_to_anchor) whenever more than
      two series are present, to guarantee no text-on-text overlap.
    * ``constrained_layout=True`` on every figure, with explicit axis
      padding so long tick labels never clip.
    * Every figure is saved as both ``<name>.png`` (300 dpi) and
      ``<name>.pdf`` (vector), with ``bbox_inches='tight'`` and a small
      uniform pad.

Inputs
------
``experiments/master_results.json`` -- the single source of truth consolidated
from Tinker runs, Modal baselines, and the legacy TRL sweep.

Outputs (written to ``paper/figures/v2/``)
------------------------------------------
1.  ``learning_curves``          -- GRPO reward trajectories for key models.
2.  ``comparison_bars``          -- final reward across experiment families.
3.  ``scaling``                  -- model-size vs last-10 reward (log-x).
4.  ``ppo_vs_grpo``              -- paired PPO vs GRPO on gsm8k.
5.  ``sensitivity_heatmap``      -- per-model x task final reward grid.
6.  ``kl_proxy``                 -- Z-value-filtered frac + loss/reward proxy.
7.  ``group_size_ablation``      -- peak/last-10 reward vs GRPO group size.
8.  ``framework_comparison``     -- Tinker-GRPO vs legacy frameworks.

Usage
-----
    python scripts/regenerate_figures.py

The script is idempotent and deterministic; a seed is fixed for the small
amount of jitter used on the scatter plots.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "experiments" / "master_results.json"
OUT_DIR = REPO_ROOT / "paper" / "figures" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style: Okabe-Ito colorblind-safe palette + serif-matched typography
# ---------------------------------------------------------------------------
# Okabe & Ito (2008): "Color Universal Design" palette. Safe for the three
# most common forms of color vision deficiency.
OKABE_ITO = {
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "black":          "#000000",
    "gray":           "#636363",  # added neutral for grid/annotations
}
PALETTE = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]

# Semantic assignments used across figures for cross-figure consistency.
METHOD_COLOR = {
    "GRPO":          OKABE_ITO["blue"],
    "Tinker GRPO":   OKABE_ITO["blue"],
    "TRL-GRPO":      OKABE_ITO["sky_blue"],
    "PPO":           OKABE_ITO["vermillion"],
    "Modal PPO":     OKABE_ITO["vermillion"],
    "PPO-REINFORCE": OKABE_ITO["vermillion"],
    "SB3 PPO":       OKABE_ITO["orange"],
    "CleanRL PPO":   OKABE_ITO["reddish_purple"],
    "Tianshou PPO":  OKABE_ITO["bluish_green"],
    "DPO":           OKABE_ITO["yellow"],
    "Old TRL":       OKABE_ITO["reddish_purple"],
    "Team Member":   OKABE_ITO["bluish_green"],
}

# Prefer Times / STIX-matching serifs, fall back gracefully.
_SERIF_STACK = ["Times New Roman", "STIXGeneral", "Liberation Serif",
                "DejaVu Serif", "serif"]

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,         # embed TrueType (editable in PDF readers)
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": _SERIF_STACK,
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "regular",
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.borderaxespad": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
    "patch.linewidth": 0.7,
})

np.random.seed(7)


# ---------------------------------------------------------------------------
# Data loading & small utilities
# ---------------------------------------------------------------------------
def load_results() -> Dict[str, Any]:
    with open(DATA_PATH, "r") as fh:
        return json.load(fh)


def get_exp_id(rec: Dict[str, Any]) -> str:
    return rec.get("experiment_id") or rec.get("experiment") or "<unknown>"


def save_fig(fig: plt.Figure, stem: str) -> None:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png.relative_to(REPO_ROOT)}")
    print(f"  wrote {pdf.relative_to(REPO_ROOT)}")


def running_mean(arr: Sequence[float], k: int = 3) -> np.ndarray:
    """Small centered moving average, edge-safe."""
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return a
    k = max(1, min(k, len(a)))
    out = np.convolve(a, np.ones(k) / k, mode="same")
    return out


PRETTY_MODEL = {
    "qwen3-8b":            "Qwen3-8B",
    "qwen3-8b-base":       "Qwen3-8B (base)",
    "qwen3-32b":           "Qwen3-32B",
    "qwen3-4b":            "Qwen3-4B",
    "qwen3.5-4b":          "Qwen3.5-4B",
    "qwen3.5-27b":         "Qwen3.5-27B",
    "qwen3-235b-moe":      "Qwen3-235B MoE",
    "qwen3-30b-moe":       "Qwen3-30B MoE",
    "qwen3-30b-moe-inst":  "Qwen3-30B MoE (Inst)",
    "llama-8b-inst":       "Llama-3.1-8B (Inst)",
    "llama-3.2-1b":        "Llama-3.2-1B",
    "llama-3.2-3b":        "Llama-3.2-3B",
    "deepseek-v3.1":       "DeepSeek-V3.1",
    "nemotron-120b":       "Nemotron-120B",
    "kimi-k2":             "Kimi-K2",
    "qwen2.5-0.5b":        "Qwen2.5-0.5B",
    "gpt-oss-20b":         "GPT-OSS-20B",
    "3b-tool":             "Qwen3-3B (tool)",
    "0.5b-dpo":            "Qwen2.5-0.5B (DPO)",
    "sb3-ppo":             "SB3 PPO",
    "cleanrl-ppo":         "CleanRL PPO",
    "tianshou-ppo":        "Tianshou PPO",
    "llama-70b":           "Llama-3.1-70B",
    "llama-3.1-8b":        "Llama-3.1-8B",
    "gpt-oss-120b":        "GPT-OSS-120B",
    "qwen35-397b":         "Qwen3.5-397B MoE",
    "kimi-k2-thinking":    "Kimi-K2 Thinking",
    "kimi-k25":            "Kimi-K2.5",
}

# Approximate dense-equivalent parameter counts (in billions) for scaling plot.
MODEL_PARAMS_B = {
    "qwen2.5-0.5b":        0.5,
    "llama-3.2-1b":        1.0,
    "llama-3.2-3b":        3.0,
    "qwen3-4b":            4.0,
    "qwen3.5-4b":          4.0,
    "qwen3-8b":            8.0,
    "llama-8b-inst":       8.0,
    "llama-3.1-8b":        8.0,
    "qwen3-32b":           32.0,
    "qwen3.5-27b":         27.0,
    "qwen3-30b-moe":       30.0,   # activated ~3B but we use total for plot
    "qwen3-30b-moe-inst":  30.0,
    "llama-70b":           70.0,
    "nemotron-120b":       120.0,
    "gpt-oss-120b":        120.0,
    "deepseek-v3.1":       671.0,  # MoE total; noted with marker style
    "qwen3-235b-moe":      235.0,
    "kimi-k2":             1000.0,
    "kimi-k2-thinking":    1000.0,
    "kimi-k25":            1000.0,
    "qwen35-397b":         397.0,
}


def pretty(m: Optional[str]) -> str:
    if not m:
        return "?"
    return PRETTY_MODEL.get(m, m)


# ---------------------------------------------------------------------------
# 1. Learning curves -- GRPO reward per step for the flagship runs
# ---------------------------------------------------------------------------
def fig1_learning_curves(data: Dict[str, Any]) -> None:
    print("[1/8] learning_curves")

    # Select a curated set of flagship GRPO runs with dense reward traces.
    keep = {
        "frontier_gsm8k_deepseek-v3.1": ("DeepSeek-V3.1 (GRPO)",  OKABE_ITO["blue"],         "-"),
        "scale_gsm8k_qwen3-8b":         ("Qwen3-8B (GRPO)",       OKABE_ITO["vermillion"],    "-"),
        "campaign_v2_w1_qwen3-8b-base": ("Qwen3-8B Base (GRPO)",  OKABE_ITO["bluish_green"],  "--"),
        "ppo_qwen3-8b":                 ("Qwen3-8B (PPO-REINFORCE)",  OKABE_ITO["orange"],   "-"),
        "ppo_llama-8b-inst":            ("Llama-3.1-8B (PPO-REINFORCE)", OKABE_ITO["reddish_purple"], "-"),
        "cross_tool_llama-8b-inst":     ("Llama-3.1-8B tool_use (GRPO)", OKABE_ITO["sky_blue"], ":"),
    }

    by_id = {get_exp_id(e): e for e in data["experiments"]}

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    for exp_id, (label, color, ls) in keep.items():
        rec = by_id.get(exp_id)
        if not rec:
            continue
        trace = rec.get("reward_trace") or []
        if not trace:
            continue
        xs = np.arange(1, len(trace) + 1)
        ys = np.asarray(trace, dtype=float)
        ax.plot(xs, ys, color=color, linestyle=ls, alpha=0.35, linewidth=1.0)
        ax.plot(xs, running_mean(ys, 3), color=color, linestyle=ls,
                linewidth=2.0, label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward (0-1, batch mean)")
    ax.set_title("GRPO vs PPO-REINFORCE learning curves (GSM8K, tool\\_use)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(left=0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # External legend on right, outside axes -- no overlap with lines.
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    borderaxespad=0.0, handlelength=2.4)
    leg.set_title("Run", prop={"size": 9, "weight": "bold"})

    save_fig(fig, "learning_curves")


# ---------------------------------------------------------------------------
# 2. Comparison bars -- final accuracy across experiment families
# ---------------------------------------------------------------------------
def fig2_comparison_bars(data: Dict[str, Any]) -> None:
    print("[2/8] comparison_bars")

    exps = data["experiments"]
    trl = data.get("trl_grpo_baseline_summary", {})

    # Aggregate into named buckets: (label, values_list, color).
    buckets: List[Tuple[str, List[float], str]] = []

    # Legacy TRL GRPO on Qwen2.5-0.5B (5 seeds)
    if trl.get("accuracies"):
        buckets.append(("TRL GRPO\n(Qwen2.5-0.5B)",
                        list(trl["accuracies"]),
                        OKABE_ITO["sky_blue"]))

    # Legacy PPO baselines (5 seeds each)
    for fw, color in (("sb3-ppo",     OKABE_ITO["orange"]),
                       ("cleanrl-ppo", OKABE_ITO["reddish_purple"]),
                       ("tianshou-ppo", OKABE_ITO["bluish_green"])):
        vals = [e.get("last10_avg") for e in exps
                if e.get("group") == "Old TRL" and e.get("model_short") == fw
                and e.get("last10_avg") is not None]
        if vals:
            label = {"sb3-ppo": "SB3 PPO",
                     "cleanrl-ppo": "CleanRL PPO",
                     "tianshou-ppo": "Tianshou PPO"}[fw]
            buckets.append((f"{label}\n(math toy)", vals, color))

    # Modal PPO-REINFORCE on GSM8K
    modal_ppo = [e.get("last10_avg") for e in exps
                 if (e.get("algorithm") == "PPO" or e.get("method") == "PPO-REINFORCE")
                 and e.get("task") == "gsm8k"
                 and e.get("last10_avg") is not None]
    if modal_ppo:
        buckets.append(("Modal PPO-REINFORCE\n(GSM8K)", modal_ppo,
                        OKABE_ITO["vermillion"]))

    # Tinker GRPO frontier / scale models on GSM8K
    tinker_grpo = [e.get("last10_avg") for e in exps
                   if e.get("method") == "GRPO" and e.get("task") == "gsm8k"
                   and (e.get("experiment_id") or "").startswith(("scale_", "frontier_", "campaign_v2_w1_"))
                   and e.get("last10_avg") is not None]
    if tinker_grpo:
        buckets.append(("Tinker GRPO\n(GSM8K frontier)", tinker_grpo,
                        OKABE_ITO["blue"]))

    # Team member runs (final peak where last10 is missing)
    team = []
    for e in exps:
        if e.get("group") == "Team Member":
            v = e.get("last10_avg") or e.get("peak_reward") or e.get("peak")
            if v is not None:
                team.append(v)
    if team:
        buckets.append(("Team members\n(multi-task)", team,
                        OKABE_ITO["black"]))

    labels = [b[0] for b in buckets]
    means = [float(np.mean(b[1])) for b in buckets]
    stds = [float(np.std(b[1], ddof=0)) for b in buckets]
    counts = [len(b[1]) for b in buckets]
    colors = [b[2] for b in buckets]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    xs = np.arange(len(labels))
    bars = ax.bar(xs, means, yerr=stds, capsize=3.5,
                  color=colors, edgecolor="#333333", linewidth=0.8,
                  error_kw={"elinewidth": 0.9, "ecolor": "#333333"})

    # Scatter individual runs on top of each bar for transparency.
    for i, b in enumerate(buckets):
        vals = np.asarray(b[1], dtype=float)
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.28
        ax.scatter(np.full_like(vals, xs[i]) + jitter, vals,
                   s=14, color="white", edgecolor="#222222",
                   linewidth=0.7, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final reward / accuracy")
    ax.set_title("Final reward by training family (mean $\\pm$ std; points = seeds/models)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # Annotate n above each bar without overlapping the error bar tip.
    for i, (m, s, n) in enumerate(zip(means, stds, counts)):
        top = min(1.04, m + s + 0.03)
        ax.text(xs[i], top, f"n={n}", ha="center", va="bottom",
                fontsize=8, color="#333333")

    # Pad x-limits so leftmost/rightmost labels don't clip.
    ax.set_xlim(-0.6, len(labels) - 0.4)

    save_fig(fig, "comparison_bars")


# ---------------------------------------------------------------------------
# 3. Scaling -- parameter count vs last-10 reward
# ---------------------------------------------------------------------------
def fig3_scaling(data: Dict[str, Any]) -> None:
    print("[3/8] scaling")

    exps = data["experiments"]

    # Build (params_B, last10, model_short, is_moe, family).
    rows = []
    for e in exps:
        eid = get_exp_id(e)
        if not any(eid.startswith(p) for p in
                   ("scale_", "frontier_", "campaign_v2_w1_", "moe_")):
            continue
        m = e.get("model_short") or ""
        # For campaign records, model_short is None -- parse from model.
        if not m:
            mm = (e.get("model") or "").lower()
            if "qwen3-8b-base" in mm: m = "qwen3-8b"
            elif "deepseek-v3.1" in mm: m = "deepseek-v3.1"
            elif "llama-3.1-70b" in mm: m = "llama-70b"
            elif "llama-3.1-8b" in mm: m = "llama-3.1-8b"
            elif "gpt-oss-120b" in mm: m = "gpt-oss-120b"
            elif "kimi-k2.5" in mm: m = "kimi-k25"
            elif "kimi-k2-thinking" in mm: m = "kimi-k2-thinking"
            elif "qwen3.5-397b" in mm: m = "qwen35-397b"
            else: m = mm or "?"
        params = MODEL_PARAMS_B.get(m)
        if params is None:
            # Derive from common naming conventions.
            if "70b" in m: params = 70.0
            elif "397b" in m: params = 397.0
            elif "120b" in m: params = 120.0
            elif "8b" in m: params = 8.0
            elif "4b" in m: params = 4.0
            elif "1b" in m: params = 1.0
        if params is None:
            continue
        y = e.get("last10_avg")
        if y is None:
            y = e.get("peak_reward") or e.get("peak")
        if y is None:
            continue
        is_moe = ("moe" in m.lower()) or ("mixtral" in m.lower()) \
            or any(tag in (e.get("model") or "").lower()
                   for tag in ("moe", "kimi", "deepseek-v3", "qwen3.5-397"))
        rows.append((params, float(y), m, is_moe))

    if not rows:
        print("  (skipped: no scaling rows)")
        return

    rows.sort(key=lambda r: r[0])
    params_arr = np.array([r[0] for r in rows])
    y_arr = np.array([r[1] for r in rows])
    moe_mask = np.array([r[3] for r in rows])
    labels = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)

    ax.scatter(params_arr[~moe_mask], y_arr[~moe_mask],
               s=60, color=OKABE_ITO["blue"], edgecolor="#222", linewidth=0.7,
               label="Dense", zorder=3)
    ax.scatter(params_arr[moe_mask], y_arr[moe_mask],
               s=64, color=OKABE_ITO["vermillion"], edgecolor="#222",
               linewidth=0.7, marker="D", label="MoE / hybrid", zorder=3)

    # Log-linear fit on dense points as a visual guide.
    dense_x = params_arr[~moe_mask]
    dense_y = y_arr[~moe_mask]
    if len(dense_x) >= 3:
        logx = np.log10(dense_x)
        coef = np.polyfit(logx, dense_y, 1)
        xs_fit = np.logspace(np.log10(dense_x.min() * 0.8),
                             np.log10(dense_x.max() * 1.2), 80)
        ys_fit = np.clip(np.polyval(coef, np.log10(xs_fit)), 0, 1)
        ax.plot(xs_fit, ys_fit, color=OKABE_ITO["gray"], linestyle="--",
                linewidth=1.4, zorder=2,
                label=f"Log-linear fit (dense, slope={coef[0]:+.2f}/decade)")

    # Curated per-point label offsets (dx pts, dy pts, ha, va). Keyed by the
    # model_short we store; keys map 1:1 to rows so there is no ambiguity.
    # This yields clean placements without any text/marker collisions.
    manual = {
        # small / mid dense
        "llama-3.2-1b":      (6,  6, "left",  "bottom"),
        "llama-3.2-3b":      (6,  6, "left",  "bottom"),
        "qwen3.5-4b":        (-8, 4, "right", "bottom"),
        "qwen3-4b":          (8,  4, "left",  "bottom"),
        "qwen3-8b":          (10, 2, "left",  "center"),   # base run high
        "qwen3-8b_low":      (8, -4, "left",  "top"),      # scale run low
        "llama-3.1-8b":      (8,  6, "left",  "bottom"),
        "llama-8b-inst":     (8,  6, "left",  "bottom"),
        "qwen3.5-27b":       (8,  6, "left",  "bottom"),
        "qwen3-30b-moe":     (8,  0, "left",  "center"),
        "qwen3-30b-moe-inst": (8,  6, "left",  "bottom"),
        "qwen3-32b":         (8, -6, "left",  "top"),
        "llama-70b":         (8,  6, "left",  "bottom"),
        "gpt-oss-120b":      (0, 10, "center","bottom"),
        "nemotron-120b":     (8,  6, "left",  "bottom"),
        "qwen3-235b-moe":    (0, 10, "center","bottom"),
        "qwen35-397b":       (8, -10, "left", "top"),
        "deepseek-v3.1":     (-8, 4, "right","bottom"),    # high one
        "deepseek-v3.1_low": (-8, 0, "right","center"),    # low one
        "kimi-k2-thinking":  (8,  6, "left",  "bottom"),
        "kimi-k25":          (8, -6, "left",  "top"),
    }
    # Split qwen3-8b (two distinct runs) and deepseek-v3.1 (two distinct runs)
    # by y-value so we pick the correct manual offset.
    for x, y, lab, is_moe in rows:
        if lab == "qwen3-8b":
            key = "qwen3-8b" if y > 0.5 else "qwen3-8b_low"
        elif lab == "deepseek-v3.1":
            key = "deepseek-v3.1" if y > 0.7 else "deepseek-v3.1_low"
        else:
            key = lab
        dx, dy, ha, va = manual.get(key, (6, 6, "left", "bottom"))
        shown = pretty(lab)
        ax.annotate(shown, (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5,
                    color="#222222", ha=ha, va=va)

    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (billions, log scale)")
    ax.set_ylabel("Last-10 reward on GSM8K")
    ax.set_title("Scaling: GRPO final reward vs model size")
    ax.set_ylim(-0.03, 1.07)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(0.3, 2500)

    ax.legend(loc="lower right", handlelength=2.2, borderaxespad=0.4)
    save_fig(fig, "scaling")


# ---------------------------------------------------------------------------
# 4. PPO vs GRPO -- paired comparison on GSM8K
# ---------------------------------------------------------------------------
def fig4_ppo_vs_grpo(data: Dict[str, Any]) -> None:
    print("[4/8] ppo_vs_grpo")

    exps = data["experiments"]
    by_id = {get_exp_id(e): e for e in exps}

    # Paired runs on GSM8K for two models.
    pairs = [
        ("Qwen3-8B",
         "scale_gsm8k_qwen3-8b",
         "ppo_qwen3-8b"),
        ("Llama-3.1-8B (Inst)",
         "scale_gsm8k_llama-8b-inst",
         "ppo_llama-8b-inst"),
    ]

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=(10.2, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]})

    # --- Left: learning curves (paired) ---
    style_grpo = dict(color=OKABE_ITO["blue"], linestyle="-")
    style_ppo  = dict(color=OKABE_ITO["vermillion"], linestyle="--")

    for label, grpo_id, ppo_id in pairs:
        for eid, style, tag in ((grpo_id, style_grpo, "GRPO"),
                                (ppo_id,  style_ppo,  "PPO")):
            rec = by_id.get(eid)
            if not rec: continue
            trace = rec.get("reward_trace") or []
            if not trace: continue
            xs = np.arange(1, len(trace) + 1)
            ys = np.asarray(trace, dtype=float)
            ax_curve.plot(xs, ys, alpha=0.28, linewidth=0.9, **style)
            marker = "o" if "Qwen" in label else "s"
            ax_curve.plot(xs, running_mean(ys, 3), linewidth=2.0,
                          marker=marker, markersize=3.5,
                          markeredgecolor="white", markeredgewidth=0.5,
                          label=f"{label} -- {tag}", **style)

    ax_curve.set_xlabel("Training step")
    ax_curve.set_ylabel("Reward")
    ax_curve.set_title("Paired GSM8K learning curves")
    ax_curve.set_ylim(-0.03, 1.05)
    ax_curve.set_xlim(left=0)
    ax_curve.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=2, handlelength=2.0)

    # --- Right: paired last-10 & peak bars ---
    model_names = [p[0] for p in pairs]
    n = len(pairs)
    xs = np.arange(n)
    width = 0.18

    def val(rec, key, fallback=None):
        if rec is None: return fallback
        v = rec.get(key)
        if v is None: v = fallback
        return v

    grpo_last = [val(by_id.get(p[1]), "last10_avg", np.nan) for p in pairs]
    ppo_last  = [val(by_id.get(p[2]), "last10_avg", np.nan) for p in pairs]
    grpo_peak = [val(by_id.get(p[1]), "peak_reward",
                     val(by_id.get(p[1]), "peak", np.nan)) for p in pairs]
    ppo_peak  = [val(by_id.get(p[2]), "peak_reward",
                     val(by_id.get(p[2]), "peak", np.nan)) for p in pairs]

    ax_bar.bar(xs - 1.5 * width, grpo_last, width,
               color=OKABE_ITO["blue"], label="GRPO last-10",
               edgecolor="#222", linewidth=0.7)
    ax_bar.bar(xs - 0.5 * width, grpo_peak, width,
               color=OKABE_ITO["blue"], alpha=0.5, hatch="//",
               label="GRPO peak", edgecolor="#222", linewidth=0.7)
    ax_bar.bar(xs + 0.5 * width, ppo_last, width,
               color=OKABE_ITO["vermillion"], label="PPO last-10",
               edgecolor="#222", linewidth=0.7)
    ax_bar.bar(xs + 1.5 * width, ppo_peak, width,
               color=OKABE_ITO["vermillion"], alpha=0.5, hatch="//",
               label="PPO peak", edgecolor="#222", linewidth=0.7)

    # Value labels above bars
    for x0, offs, vals in [
        (xs, -1.5 * width, grpo_last), (xs, -0.5 * width, grpo_peak),
        (xs,  0.5 * width, ppo_last),  (xs,  1.5 * width, ppo_peak),
    ]:
        for xi, vi in zip(x0, vals):
            if vi is None or (isinstance(vi, float) and math.isnan(vi)):
                continue
            ax_bar.text(xi + offs, vi + 0.02, f"{vi:.2f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color="#222222")

    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(model_names)
    ax_bar.set_ylabel("Reward")
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_yticks(np.arange(0, 1.01, 0.2))
    ax_bar.set_title("Peak vs last-10 reward (GSM8K)")
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=2, handlelength=1.8, columnspacing=1.0)

    # The constrained_layout engine reserves space automatically for the
    # suptitle when we omit an explicit y coordinate.
    fig.suptitle("PPO-REINFORCE vs GRPO on GSM8K",
                 fontsize=12, fontweight="bold")

    save_fig(fig, "ppo_vs_grpo")


# ---------------------------------------------------------------------------
# 5. Sensitivity heatmap -- model x task final reward
# ---------------------------------------------------------------------------
def fig5_heatmap(data: Dict[str, Any]) -> None:
    print("[5/8] sensitivity_heatmap")

    exps = data["experiments"]

    # Build matrix over (model, task) using all available runs. We collapse
    # tool_calling / tool_use into a single "Tool use" column and
    # humaneval / code_generation into a single "Code" column to reduce
    # matrix sparsity while staying faithful to the raw task taxonomy.
    candidate_models = [
        "qwen3-8b", "llama-8b-inst", "qwen3.5-4b", "qwen3-4b",
        "qwen3-32b", "qwen3.5-27b", "deepseek-v3.1",
        "qwen3-30b-moe-inst", "qwen3-30b-moe", "nemotron-120b",
        "qwen3-235b-moe", "3b-tool",
    ]
    task_map = {
        "gsm8k":           "GSM8K",
        "math":            "Math\n(toy)",
        "math_reasoning":  "Math\nreasoning",
        "tool_use":        "Tool\nuse",
        "tool_calling":    "Tool\nuse",
        "code_generation": "Code\ngeneration",
        "humaneval":       "Code\ngeneration",
    }
    col_order = ["GSM8K", "Math\nreasoning", "Tool\nuse", "Code\ngeneration"]

    grid = np.full((len(candidate_models), len(col_order)), np.nan)

    for e in exps:
        m = e.get("model_short")
        t = e.get("task")
        if m is None or t is None:
            continue
        if m not in candidate_models or t not in task_map:
            continue
        col = task_map[t]
        if col not in col_order:
            continue
        v = e.get("last10_avg")
        if v is None:
            v = e.get("peak_reward") or e.get("peak")
        if v is None:
            continue
        i = candidate_models.index(m)
        j = col_order.index(col)
        # Use the max when multiple runs collide (robust to retries).
        cur = grid[i, j]
        grid[i, j] = float(v) if math.isnan(cur) else max(cur, float(v))

    # Drop all-NaN rows to avoid empty heatmap lines.
    keep_rows = [i for i in range(len(candidate_models))
                 if not np.all(np.isnan(grid[i]))]
    grid = grid[keep_rows]
    row_labels = [pretty(candidate_models[i]) for i in keep_rows]
    col_display = col_order

    fig, ax = plt.subplots(figsize=(7.4, 0.45 * len(row_labels) + 2.2),
                           constrained_layout=True)

    # Use a colorblind-safe sequential ramp (Okabe blue -> white -> vermillion).
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "okabe_div",
        [OKABE_ITO["vermillion"], "#f7f7f7", OKABE_ITO["blue"]],
        N=256)

    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(np.arange(len(col_display)))
    ax.set_xticklabels(col_display)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Final reward by model x task (Tinker GRPO; max over runs)")

    # Annotate each cell. Choose text color based on luminance.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if math.isnan(v):
                ax.text(j, i, "--", ha="center", va="center",
                        color="#999999", fontsize=8)
                continue
            txt_color = "white" if (v > 0.65 or v < 0.18) else "#111111"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    # Outside colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Reward (0 = failure, 1 = solved)")

    # Hide the internal grid for a clean heatmap
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, len(col_display), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.2)

    save_fig(fig, "sensitivity_heatmap")


# ---------------------------------------------------------------------------
# 6. KL proxy -- Z-value-filtered fraction and loss/reward together
# ---------------------------------------------------------------------------
def fig6_kl_proxy(data: Dict[str, Any]) -> None:
    print("[6/8] kl_proxy")

    exps = data["experiments"]

    # Runs carrying step_log with ``zvf`` (zero-value-filtered) fraction and loss.
    keep = {
        "scale_gsm8k_qwen3-8b":        ("Qwen3-8B (GSM8K)",   OKABE_ITO["blue"]),
        "scale_gsm8k_llama-8b-inst":   ("Llama-3.1-8B (GSM8K)", OKABE_ITO["vermillion"]),
        "scale_gsm8k_qwen3.5-4b":      ("Qwen3.5-4B (GSM8K)", OKABE_ITO["bluish_green"]),
        "frontier_gsm8k_deepseek-v3.1": ("DeepSeek-V3.1 (GSM8K)", OKABE_ITO["orange"]),
        "frontier_gsm8k_nemotron-120b": ("Nemotron-120B (GSM8K)", OKABE_ITO["reddish_purple"]),
        "cross_tool_llama-8b-inst":    ("Llama-3.1-8B (tool\\_use)", OKABE_ITO["sky_blue"]),
    }

    by_id = {get_exp_id(e): e for e in exps}

    fig, (ax_zvf, ax_loss) = plt.subplots(
        1, 2, figsize=(10.0, 4.0), constrained_layout=True)

    handles, labels = [], []
    for exp_id, (label, color) in keep.items():
        rec = by_id.get(exp_id)
        if not rec: continue
        sl = rec.get("step_log") or []
        if not sl: continue
        steps = [s["step"] for s in sl]
        zvf   = [s.get("zvf", np.nan) for s in sl]
        loss  = [s.get("loss", np.nan) for s in sl]

        (h,) = ax_zvf.plot(steps, zvf, color=color, linewidth=1.7,
                           marker="o", markersize=3.3, markeredgewidth=0)
        ax_loss.plot(steps, loss, color=color, linewidth=1.7,
                     marker="s", markersize=3.3, markeredgewidth=0)
        handles.append(h); labels.append(label)

    ax_zvf.set_xlabel("Training step")
    ax_zvf.set_ylabel("Zero-value-filtered fraction")
    ax_zvf.set_title("KL proxy: group-relative advantage collapse")
    ax_zvf.set_ylim(-0.03, 1.05)

    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Training loss (importance-weighted)")
    ax_loss.set_title("Surrogate loss trajectory")

    # Shared legend BELOW both panels. Using loc="outside lower center" lets
    # matplotlib's constrained_layout engine reserve space for it automatically
    # so it never overlaps the x-axis labels.
    fig.legend(handles, labels, loc="outside lower center",
               ncol=3, handlelength=2.0, frameon=False,
               borderaxespad=0.4)
    fig.suptitle("KL / update-magnitude proxies (ZVF fraction + surrogate loss)",
                 fontsize=12, fontweight="bold")

    save_fig(fig, "kl_proxy")


# ---------------------------------------------------------------------------
# 7. Group-size ablation -- GRPO G in {2,4,8,16,32} for Qwen3-8B
# ---------------------------------------------------------------------------
def fig7_group_size(data: Dict[str, Any]) -> None:
    print("[7/8] group_size_ablation")

    exps = data["experiments"]

    # Collect w2 ablation runs (G in {2,4,16,32}) plus w1 base (G=8).
    rows = []  # (G, peak, last10, source)
    for e in exps:
        eid = get_exp_id(e)
        if "campaign_v2_w2_qwen3-8b_G" in eid:
            try:
                G = int(eid.rsplit("_G", 1)[1])
            except ValueError:
                continue
            rows.append((G, e.get("peak_reward"), e.get("last10_avg"), "w2"))
        elif eid == "campaign_v2_w1_qwen3-8b-base":
            rows.append((8, e.get("peak_reward"), e.get("last10_avg"), "w1"))
        elif eid == "scale_gsm8k_qwen3-8b":
            rows.append((8, e.get("peak_reward"), e.get("last10_avg"), "w1"))
    rows.sort(key=lambda r: (r[0], r[3]))

    if not rows:
        print("  (skipped: no group-size runs)")
        return

    # Aggregate duplicates at a given G by taking the mean (keeps std for error).
    agg: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"peak": [], "last10": []})
    for G, p, l, _ in rows:
        if p is not None: agg[G]["peak"].append(float(p))
        if l is not None: agg[G]["last10"].append(float(l))

    Gs = sorted(agg.keys())
    peak_mean = [float(np.mean(agg[g]["peak"])) if agg[g]["peak"] else np.nan for g in Gs]
    peak_err  = [float(np.std(agg[g]["peak"], ddof=0)) if len(agg[g]["peak"]) > 1 else 0.0 for g in Gs]
    last_mean = [float(np.mean(agg[g]["last10"])) if agg[g]["last10"] else np.nan for g in Gs]
    last_err  = [float(np.std(agg[g]["last10"], ddof=0)) if len(agg[g]["last10"]) > 1 else 0.0 for g in Gs]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    xs = np.arange(len(Gs))
    width = 0.36

    b1 = ax.bar(xs - width / 2, peak_mean, width, yerr=peak_err,
                color=OKABE_ITO["blue"], edgecolor="#222", linewidth=0.7,
                label="Peak reward", capsize=3.0,
                error_kw={"elinewidth": 0.8, "ecolor": "#333"})
    b2 = ax.bar(xs + width / 2, last_mean, width, yerr=last_err,
                color=OKABE_ITO["vermillion"], edgecolor="#222", linewidth=0.7,
                label="Last-10 reward", capsize=3.0,
                error_kw={"elinewidth": 0.8, "ecolor": "#333"})

    for bars, vals in [(b1, peak_mean), (b2, last_mean)]:
        for rect, v in zip(bars, vals):
            if v is None or math.isnan(v): continue
            ax.text(rect.get_x() + rect.get_width() / 2,
                    v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#222")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"G = {g}" for g in Gs])
    ax.set_ylabel("Reward on GSM8K")
    ax.set_title("GRPO group-size ablation (Qwen3-8B, GSM8K, 30 steps)")
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.4)

    # Add an annotation for the "sweet spot" -- place the label in empty space
    # above the G=2 / G=4 bars where there is no bar or error bar to collide with.
    try:
        best_idx = int(np.nanargmax(last_mean))
        ax.annotate("sweet spot (best last-10)",
                    xy=(xs[best_idx] + width / 2, last_mean[best_idx]),
                    xytext=(xs[best_idx] + width / 2 - 0.9,
                            last_mean[best_idx] + 0.45),
                    ha="center",
                    fontsize=8.5, color="#444",
                    arrowprops=dict(arrowstyle="->", color="#444",
                                    lw=0.8, shrinkA=2, shrinkB=4,
                                    connectionstyle="arc3,rad=-0.12"))
    except ValueError:
        pass

    save_fig(fig, "group_size_ablation")


# ---------------------------------------------------------------------------
# 8. Framework comparison -- Tinker GRPO vs legacy RL frameworks
# ---------------------------------------------------------------------------
def fig8_framework_comparison(data: Dict[str, Any]) -> None:
    print("[8/8] framework_comparison")

    exps = data["experiments"]
    trl = data.get("trl_grpo_baseline_summary", {})

    frameworks: List[Tuple[str, List[float], str]] = []

    # Legacy TRL GRPO
    if trl.get("accuracies"):
        frameworks.append(("TRL GRPO\n(Qwen2.5-0.5B, math)",
                           list(trl["accuracies"]),
                           OKABE_ITO["sky_blue"]))

    for fw, color, label in (
            ("sb3-ppo",     OKABE_ITO["orange"],         "Stable-Baselines3 PPO\n(math toy)"),
            ("cleanrl-ppo", OKABE_ITO["reddish_purple"], "CleanRL PPO\n(math toy)"),
            ("tianshou-ppo", OKABE_ITO["bluish_green"],  "Tianshou PPO\n(math toy)")):
        vals = [e.get("last10_avg") for e in exps
                if e.get("group") == "Old TRL"
                and e.get("model_short") == fw
                and e.get("last10_avg") is not None]
        if vals:
            frameworks.append((label, vals, color))

    # Modal PPO-REINFORCE on real LLMs
    modal = [e.get("last10_avg") for e in exps
             if (e.get("algorithm") == "PPO" or e.get("method") == "PPO-REINFORCE")
             and e.get("task") == "gsm8k" and e.get("last10_avg") is not None]
    if modal:
        frameworks.append(("Modal PPO-REINFORCE\n(8B LLMs, GSM8K)", modal,
                           OKABE_ITO["vermillion"]))

    # Tinker GRPO on frontier LLMs
    tinker = [e.get("last10_avg") for e in exps
              if e.get("method") == "GRPO" and e.get("task") == "gsm8k"
              and (e.get("experiment_id") or "").startswith(("scale_", "frontier_", "campaign_v2_w1_"))
              and e.get("last10_avg") is not None]
    if tinker:
        frameworks.append(("Tinker GRPO\n(frontier LLMs, GSM8K)", tinker,
                           OKABE_ITO["blue"]))

    # Sort frameworks by mean for readability.
    frameworks.sort(key=lambda fw: np.mean(fw[1]))

    labels = [f[0] for f in frameworks]
    values = [f[1] for f in frameworks]
    colors = [f[2] for f in frameworks]
    means = [float(np.mean(v)) for v in values]
    stds = [float(np.std(v, ddof=0)) for v in values]
    counts = [len(v) for v in values]

    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(labels) + 2.6),
                           constrained_layout=True)
    ys = np.arange(len(labels))

    ax.barh(ys, means, xerr=stds, color=colors,
            edgecolor="#222", linewidth=0.8, capsize=3.0,
            error_kw={"elinewidth": 0.9, "ecolor": "#333"})

    # Strip plot of individual runs
    for i, v in enumerate(values):
        arr = np.asarray(v)
        jitter = (np.random.rand(len(arr)) - 0.5) * 0.35
        ax.scatter(arr, np.full_like(arr, ys[i], dtype=float) + jitter,
                   s=18, color="white", edgecolor="#222", linewidth=0.6,
                   zorder=3)

    # Put numeric annotations in a dedicated right margin at a fixed x so the
    # text never overlaps the scatter points or the bar error-bar caps.
    x_text = 1.22
    for i, (m, s, n) in enumerate(zip(means, stds, counts)):
        ax.text(x_text, ys[i],
                f"{m:.3f} $\\pm$ {s:.3f}  (n={n})",
                va="center", ha="left", fontsize=8.5, color="#222",
                clip_on=False)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.20)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Final reward (last-10 mean)")
    ax.set_title("Framework comparison on reinforcement fine-tuning")
    ax.invert_yaxis()

    # Subtle note about the scale gap
    ax.axvline(0.5, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=1)

    save_fig(fig, "framework_comparison")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    if not DATA_PATH.exists():
        print(f"FATAL: {DATA_PATH} not found.", file=sys.stderr)
        return 1
    data = load_results()
    print(f"Loaded {len(data['experiments'])} experiment records from "
          f"{DATA_PATH.relative_to(REPO_ROOT)}")
    print(f"Writing figures to {OUT_DIR.relative_to(REPO_ROOT)}")
    print()

    fig1_learning_curves(data)
    fig2_comparison_bars(data)
    fig3_scaling(data)
    fig4_ppo_vs_grpo(data)
    fig5_heatmap(data)
    fig6_kl_proxy(data)
    fig7_group_size(data)
    fig8_framework_comparison(data)

    print()
    print("Done. Figures regenerated at 300 dpi PNG + vector PDF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
