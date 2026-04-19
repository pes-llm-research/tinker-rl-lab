"""Generate the Task 4 4-bar figure: Tinker vs TRL vs verl vs OpenRLHF.

Reads ``experiments/results/framework_comparison.json`` and writes
``experiments/results/framework_comparison.png`` (and .pdf). Bars whose ``mode``
is ``dryrun`` are rendered with a hatched pattern and an ``[awaiting H100]``
annotation so they can't be mistaken for real runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parent / "framework_comparison.json"
PNG = RESULTS.with_suffix(".png")
PDF = RESULTS.with_suffix(".pdf")

COLORS = {
    "Tinker": "#1f77b4",
    "TRL": "#ff7f0e",
    "verl": "#2ca02c",
    "OpenRLHF": "#9467bd",
}


def main() -> None:
    data = json.loads(RESULTS.read_text())
    frameworks = data["frameworks"]
    names = [f["framework"] for f in frameworks]
    last10 = [float(f["last10_avg"]) for f in frameworks]
    modes = [f.get("mode", "real") for f in frameworks]

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    x = np.arange(len(names))
    bars = []
    for i, (name, val, mode) in enumerate(zip(names, last10, modes)):
        color = COLORS.get(name, "#555")
        hatch = "////" if mode == "dryrun" else None
        alpha = 0.55 if mode == "dryrun" else 0.95
        edge = "#222"
        b = ax.bar(
            x[i], val, width=0.62,
            color=color, alpha=alpha,
            edgecolor=edge, linewidth=1.0, hatch=hatch,
        )
        bars.append(b)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Last-10 reward (GSM8K-500, G=8, lr=1e-5, 30 steps)", fontsize=10)
    ax.set_title("Framework gap: Qwen3-8B GRPO, identical config across launchers",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(1.0, max(last10) * 1.25))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Value labels
    for xi, val, mode in zip(x, last10, modes):
        label = f"{val:.3f}"
        if mode == "dryrun":
            label += "\n[awaiting H100]"
        ax.text(xi, val + 0.02, label, ha="center", va="bottom", fontsize=9,
                color="#222", fontweight="bold" if mode == "real" else "normal")

    # Footer / legend
    has_dry = any(m == "dryrun" for m in modes)
    caption = "Real Modal-H100 runs = solid bars."
    if has_dry:
        caption += "  Hatched bars = seeded-deterministic dryrun (overwrite via modal_grpo_{verl,openrlhf}.py)."
    fig.text(0.5, -0.02, caption, ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    fig.savefig(PNG, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    print(f"Wrote {PNG}")
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
