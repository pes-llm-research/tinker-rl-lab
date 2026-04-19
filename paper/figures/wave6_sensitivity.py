#!/usr/bin/env python3
"""
Wave 6 sensitivity figure generator.

Produces a 3-panel figure for the Qwen3-8B GSM8K ablations:
  • Panel 1 — Sampling temperature vs. peak / last-10 reward
  • Panel 2 — LoRA rank vs. peak / last-10 reward
  • Panel 3 — Per-step batch size vs. peak / last-10 reward

Reads: experiments/tinker-runs/results/wave6_ablations.json
Writes: paper/figures/wave6_sensitivity.png (+ matching .pdf)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Default paths are derived from this file's location so the script works
# from any checkout. Override via --results / --out.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
RESULTS_PATH = os.path.join(
    _REPO_ROOT,
    "experiments",
    "tinker-runs",
    "results",
    "wave6_ablations.json",
)
OUT_PNG = os.path.join(_HERE, "wave6_sensitivity.png")
OUT_PDF = OUT_PNG.replace(".png", ".pdf")


def load():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _completed(rows, key):
    xs, peaks, last10s, first5s = [], [], [], []
    for r in rows:
        if r.get("status") != "completed":
            continue
        xs.append(r[key])
        peaks.append(r.get("peak_reward", r.get("peak")))
        last10s.append(r.get("last10_avg"))
        first5s.append(r.get("first5_avg"))
    order = np.argsort(xs)
    return (
        np.array(xs)[order],
        np.array(peaks, dtype=float)[order],
        np.array(last10s, dtype=float)[order],
        np.array(first5s, dtype=float)[order],
    )


def _panel(ax, xs, peaks, last10s, first5s, title, xlabel, log_x=False, xticks=None):
    if log_x:
        ax.set_xscale("log", base=2)
    ax.plot(
        xs,
        peaks,
        marker="o",
        lw=2.0,
        ms=8,
        color="#d6604d",
        label="Peak reward",
    )
    ax.plot(
        xs,
        last10s,
        marker="s",
        lw=2.0,
        ms=7,
        color="#2166ac",
        label="Last-10 avg",
    )
    ax.plot(
        xs,
        first5s,
        marker="^",
        lw=1.5,
        ms=6,
        color="#4dac26",
        alpha=0.75,
        label="First-5 avg",
    )
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("GSM8K training reward")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(data, out_png=OUT_PNG, out_pdf=OUT_PDF):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Temperature
    xs, pk, l10, f5 = _completed(data["temperature_sweep"], "temperature")
    _panel(
        axes[0],
        xs,
        pk,
        l10,
        f5,
        "Sampling temperature sensitivity",
        "Sampling temperature",
        log_x=False,
        xticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    )

    # Rank
    xs, pk, l10, f5 = _completed(data["rank_sweep"], "rank")
    _panel(
        axes[1],
        xs,
        pk,
        l10,
        f5,
        "LoRA rank sensitivity",
        "LoRA rank",
        log_x=True,
        xticks=[4, 8, 16, 32, 64],
    )

    # Batch
    xs, pk, l10, f5 = _completed(data["batch_sweep"], "batch")
    _panel(
        axes[2],
        xs,
        pk,
        l10,
        f5,
        "Per-step batch size sensitivity",
        "Prompts per GRPO step",
        log_x=True,
        xticks=[1, 2, 4, 8],
    )

    axes[0].legend(loc="lower right", frameon=False)

    md = data.get("metadata", {})
    sup = (
        f"Qwen3-8B · GSM8K · seed {md.get('seed', 42)} · "
        f"steps {md.get('steps', 30)} · group G={md.get('group_size', 8)} · "
        f"lr {md.get('lr', '1e-5')}   |   baseline: "
        f"rank {md.get('baseline', {}).get('rank')}, "
        f"T={md.get('baseline', {}).get('temperature')}, "
        f"batch {md.get('baseline', {}).get('batch')}"
    )
    fig.suptitle(
        "Wave 6: Qwen3-8B GRPO sensitivity to temperature, LoRA rank, and batch size",
        fontsize=14,
        y=1.02,
        fontweight="bold",
    )
    fig.text(0.5, -0.02, sup, ha="center", fontsize=9.5, color="#555")

    plt.tight_layout()

    png_dir = os.path.dirname(out_png)
    pdf_dir = os.path.dirname(out_pdf)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
    if pdf_dir and pdf_dir != png_dir:
        os.makedirs(pdf_dir, exist_ok=True)

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=RESULTS_PATH)
    p.add_argument("--out", default=OUT_PNG)
    args = p.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    make_figure(data, args.out, args.out.replace(".png", ".pdf"))
