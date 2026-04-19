"""
Generate all figures for the TinkerRL NeurIPS paper.
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────
with open("/home/user/workspace/tinker-rl-lab/experiments/all_results_consolidated.json") as f:
    data = json.load(f)

# ─────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.7,
    }
)

OUTDIR = "/home/user/workspace/tinker-rl-lab/paper/figures/"

# ─────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────
# Model family → base color
FAMILY_COLORS = {
    "qwen": "#2166ac",  # blue
    "llama": "#d6604d",  # red
    "deepseek": "#4dac26",  # green
    "nemotron": "#f4a582",  # orange
    "moe": "#762a83",  # purple
}

MODEL_COLORS = {
    "qwen3-8b": "#084594",
    "qwen3.5-4b": "#2171b5",
    "qwen3.5-27b": "#4292c6",
    "qwen3-32b": "#6baed6",
    "qwen3-235b-moe": "#9ecae1",
    "qwen3-30b-moe-inst": "#c6dbef",
    "qwen3-30b-moe": "#deebf7",
    "llama-8b-inst": "#d6604d",
    "deepseek-v3.1": "#4dac26",
    "nemotron-120b": "#f4a582",
}

# ─────────────────────────────────────────────────────────
# Helper: pretty model names
# ─────────────────────────────────────────────────────────
PRETTY = {
    "qwen3-8b": "Qwen3-8B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3.5-27b": "Qwen3.5-27B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-235b-moe": "Qwen3-235B (MoE)",
    "qwen3-30b-moe-inst": "Qwen3-30B-Inst (MoE)",
    "qwen3-30b-moe": "Qwen3-30B (MoE)",
    "llama-8b-inst": "Llama-3.1-8B",
    "deepseek-v3.1": "DeepSeek-V3.1",
    "nemotron-120b": "Nemotron-120B",
}


def pname(ms):
    return PRETTY.get(ms, ms)


def is_partial(exp):
    return exp.get("partial", False)


# ═══════════════════════════════════════════════════════════
# Figure 1: Learning curves – all GSM8K GRPO experiments
# ═══════════════════════════════════════════════════════════
gsm8k_exps = [
    e
    for e in data
    if e.get("task") == "gsm8k"
    and e.get("platform", "") != "modal_h100"
    and e.get("algorithm", "") != "PPO"
]

fig, ax = plt.subplots(figsize=(10, 6))

for exp in gsm8k_exps:
    trace = exp.get("reward_trace", [])
    if not trace:
        continue
    ms = exp["model_short"]
    color = MODEL_COLORS.get(ms, "#888888")
    steps = list(range(1, len(trace) + 1))
    lw = 1.8
    label = pname(ms)
    # smooth with rolling average (window=3)
    sm = np.convolve(trace, np.ones(3) / 3, mode="valid")
    sm_steps = steps[1 : len(sm) + 1]
    if is_partial(exp):
        ax.plot(steps, trace, color=color, linewidth=lw, linestyle="--", alpha=0.5)
        ax.plot(
            sm_steps,
            sm,
            color=color,
            linewidth=lw + 0.5,
            linestyle="--",
            alpha=0.9,
            label=label + " (partial)",
        )
    else:
        ax.plot(steps, trace, color=color, linewidth=lw, linestyle="-", alpha=0.35)
        ax.plot(
            sm_steps, sm, color=color, linewidth=lw + 0.5, linestyle="-", alpha=0.9, label=label
        )

ax.set_xlabel("Training Step")
ax.set_ylabel("Mean Reward")
ax.set_title("GRPO Learning Curves on GSM8K")
ax.set_ylim(-0.05, 1.15)
ax.set_xlim(0.5)

# Legend – deduplicate
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left", framealpha=0.9, ncol=2)

# Add dashed/solid legend annotation
solid_line = Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="Full run")
dash_line = Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Partial run")
ax.add_artist(
    ax.legend(
        handles=list(by_label.values()) + [solid_line, dash_line],
        labels=list(by_label.keys()) + ["Full run", "Partial run"],
        loc="upper left",
        framealpha=0.9,
        ncol=2,
        fontsize=9,
    )
)

plt.tight_layout()
plt.savefig(OUTDIR + "learning_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ learning_curves.png")

# ═══════════════════════════════════════════════════════════
# Figure 2: Grouped bar chart – peak vs last-10 accuracy
# ═══════════════════════════════════════════════════════════
# Gather one entry per model (prefer full run if duplicate)
model_data = {}
for exp in data:
    if exp.get("task") != "gsm8k":
        continue
    ms = exp["model_short"]
    algo = exp.get("algorithm", "GRPO")
    platform = exp.get("platform", "tinker")
    key = ms + ("_ppo" if algo == "PPO" else "_grpo")

    peak = exp.get("peak", exp.get("peak_accuracy", 0)) or 0
    last10 = exp.get("last10_avg", exp.get("last10_accuracy", 0)) or 0
    partial = is_partial(exp)

    if key not in model_data or peak > model_data[key]["peak"]:
        model_data[key] = {
            "model_short": ms,
            "label": pname(ms) + (" [PPO]" if algo == "PPO" else " [GRPO]"),
            "peak": peak,
            "last10": last10,
            "partial": partial,
            "algo": algo,
        }

# sort by peak desc
rows = sorted(model_data.values(), key=lambda r: -r["peak"])

labels = [r["label"] for r in rows]
peaks = [r["peak"] * 100 for r in rows]
last10s = [r["last10"] * 100 for r in rows]
partials = [r["partial"] for r in rows]
algos = [r["algo"] for r in rows]

x = np.arange(len(rows))
bar_w = 0.35

fig, ax = plt.subplots(figsize=(12, 5))

for i, (pk, l10, part, algo) in enumerate(zip(peaks, last10s, partials, algos)):
    color_peak = "#1a5276" if algo == "GRPO" else "#78281f"
    color_last10 = "#5dade2" if algo == "GRPO" else "#e74c3c"
    hatch = "//" if part else ""
    ax.bar(
        x[i] - bar_w / 2,
        pk,
        bar_w,
        color=color_peak,
        hatch=hatch,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.bar(
        x[i] + bar_w / 2,
        l10,
        bar_w,
        color=color_last10,
        hatch=hatch,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        alpha=0.85,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Peak vs Last-10 Accuracy on GSM8K")
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

legend_handles = [
    mpatches.Patch(color="#1a5276", label="GRPO – Peak"),
    mpatches.Patch(color="#5dade2", label="GRPO – Last-10"),
    mpatches.Patch(color="#78281f", label="PPO – Peak"),
    mpatches.Patch(color="#e74c3c", label="PPO – Last-10"),
    mpatches.Patch(facecolor="white", edgecolor="gray", hatch="//", label="Partial run"),
]
ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig(OUTDIR + "comparison_bars.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ comparison_bars.png")

# ═══════════════════════════════════════════════════════════
# Figure 3: Scaling plot – model size vs peak accuracy
# ═══════════════════════════════════════════════════════════
MODEL_SIZES = {
    "qwen3-8b": 8,
    "qwen3.5-4b": 4,
    "qwen3.5-27b": 27,
    "qwen3-32b": 32,
    "qwen3-235b-moe": 22,  # active params
    "llama-8b-inst": 8,
    "deepseek-v3.1": 685,
    "nemotron-120b": 12,  # active
    "qwen3-30b-moe-inst": 3,  # A3B active
    "qwen3-30b-moe": 3,
}

# Build per-model best peak
model_peak = {}
for exp in data:
    if exp.get("task") != "gsm8k":
        continue
    if exp.get("algorithm", "") == "PPO":
        continue
    ms = exp["model_short"]
    pk = exp.get("peak", 0) or 0
    if ms not in model_peak or pk > model_peak[ms]:
        model_peak[ms] = pk

# Qwen scaling series (dense models or MoE as labelled in task)
qwen_series = {k: v for k, v in model_peak.items() if "qwen" in k and k != "qwen3-235b-moe"}
other_series = {k: v for k, v in model_peak.items() if k not in qwen_series}

fig, ax = plt.subplots(figsize=(8, 5))

# Qwen points + trend line
qx = np.array([MODEL_SIZES[k] for k in qwen_series if k in MODEL_SIZES])
qy = np.array([v * 100 for k, v in qwen_series.items() if k in MODEL_SIZES])
# sort by size
sort_idx = np.argsort(qx)
qx, qy = qx[sort_idx], qy[sort_idx]

# trend on log scale
if len(qx) >= 2:
    log_qx = np.log10(qx)
    slope, intercept, r, p, se = stats.linregress(log_qx, qy)
    x_fit = np.logspace(np.log10(min(qx) * 0.8), np.log10(max(qx) * 1.2), 200)
    y_fit = slope * np.log10(x_fit) + intercept
    ax.plot(
        x_fit,
        y_fit,
        color=FAMILY_COLORS["qwen"],
        linewidth=1.2,
        linestyle="--",
        alpha=0.6,
        label="Qwen trend",
    )

for k, v in qwen_series.items():
    if k not in MODEL_SIZES:
        continue
    sz = MODEL_SIZES[k]
    pk = v * 100
    color = MODEL_COLORS.get(k, FAMILY_COLORS["qwen"])
    ax.scatter(sz, pk, color=color, s=120, zorder=5, edgecolors="white", linewidths=0.8)
    ax.annotate(
        pname(k), (sz, pk), textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=color
    )

# MoE Qwen235B
k = "qwen3-235b-moe"
if k in model_peak and k in MODEL_SIZES:
    sz = MODEL_SIZES[k]
    pk = model_peak[k] * 100
    ax.scatter(
        sz,
        pk,
        color=FAMILY_COLORS["moe"],
        marker="D",
        s=130,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        pname(k),
        (sz, pk),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8.5,
        color=FAMILY_COLORS["moe"],
    )

# Llama
k = "llama-8b-inst"
if k in model_peak and k in MODEL_SIZES:
    sz = MODEL_SIZES[k]
    pk = model_peak[k] * 100
    ax.scatter(
        sz,
        pk,
        color=FAMILY_COLORS["llama"],
        marker="s",
        s=130,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        pname(k),
        (sz, pk),
        textcoords="offset points",
        xytext=(6, -12),
        fontsize=8.5,
        color=FAMILY_COLORS["llama"],
    )

# DeepSeek
k = "deepseek-v3.1"
if k in model_peak and k in MODEL_SIZES:
    sz = MODEL_SIZES[k]
    pk = model_peak[k] * 100
    ax.scatter(
        sz,
        pk,
        color=FAMILY_COLORS["deepseek"],
        marker="^",
        s=140,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        pname(k),
        (sz, pk),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8.5,
        color=FAMILY_COLORS["deepseek"],
    )

# Nemotron
k = "nemotron-120b"
if k in model_peak and k in MODEL_SIZES:
    sz = MODEL_SIZES[k]
    pk = model_peak[k] * 100
    ax.scatter(
        sz,
        pk,
        color=FAMILY_COLORS["nemotron"],
        marker="P",
        s=130,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        pname(k),
        (sz, pk),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8.5,
        color=FAMILY_COLORS["nemotron"],
    )

ax.set_xscale("log")
ax.set_xlabel("Model Size (B active params, log scale)")
ax.set_ylabel("Peak Accuracy (%)")
ax.set_title("Scaling: Model Size vs Peak GSM8K Accuracy")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(0, 115)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=FAMILY_COLORS["qwen"],
        markersize=9,
        label="Qwen (dense)",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor=FAMILY_COLORS["moe"],
        markersize=9,
        label="Qwen (MoE)",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=FAMILY_COLORS["llama"],
        markersize=9,
        label="Llama",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor=FAMILY_COLORS["deepseek"],
        markersize=9,
        label="DeepSeek",
    ),
    Line2D(
        [0],
        [0],
        marker="P",
        color="w",
        markerfacecolor=FAMILY_COLORS["nemotron"],
        markersize=9,
        label="Nemotron",
    ),
]
ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTDIR + "scaling_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ scaling_plot.png")

# ═══════════════════════════════════════════════════════════
# Figure 4: PPO vs GRPO side-by-side comparison
# ═══════════════════════════════════════════════════════════
# From the data + task spec hardcoded values:
ppo_grpo = {
    "Qwen3-8B": {
        "PPO": {"peak": 75.0, "last10": 22.5},
        "GRPO": {"peak": 62.5, "last10": 34.375},
    },
    "Llama-3.1-8B": {
        "PPO": {"peak": 100.0, "last10": 97.5},
        "GRPO": {"peak": 100.0, "last10": 84.375},
    },
}

# Override with data where available
for exp in data:
    if exp.get("task") != "gsm8k":
        continue
    ms = exp["model_short"]
    algo = exp.get("algorithm", "GRPO")
    pk = (exp.get("peak", 0) or 0) * 100
    l10 = (exp.get("last10_avg", 0) or 0) * 100
    label_map = {"qwen3-8b": "Qwen3-8B", "llama-8b-inst": "Llama-3.1-8B"}
    if ms in label_map:
        mdl = label_map[ms]
        if mdl in ppo_grpo and algo in ppo_grpo[mdl]:
            if pk > 0:
                ppo_grpo[mdl][algo]["peak"] = pk
                ppo_grpo[mdl][algo]["last10"] = l10

models = list(ppo_grpo.keys())
algos = ["PPO", "GRPO"]
colors_peak = {"PPO": "#78281f", "GRPO": "#1a5276"}
colors_last10 = {"PPO": "#e74c3c", "GRPO": "#5dade2"}

n_models = len(models)
group_w = 1.0
bar_w = 0.18
offsets = {
    "PPO_peak": -1.5 * bar_w,
    "PPO_last10": -0.5 * bar_w,
    "GRPO_peak": 0.5 * bar_w,
    "GRPO_last10": 1.5 * bar_w,
}

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(n_models) * group_w

for i, mdl in enumerate(models):
    for algo in algos:
        vals = ppo_grpo[mdl][algo]
        # peak bar
        key_pk = f"{algo}_peak"
        ax.bar(
            x[i] + offsets[key_pk],
            vals["peak"],
            bar_w,
            color=colors_peak[algo],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        # last10 bar
        key_l10 = f"{algo}_last10"
        ax.bar(
            x[i] + offsets[key_l10],
            vals["last10"],
            bar_w,
            color=colors_last10[algo],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            alpha=0.85,
        )

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Accuracy (%)")
ax.set_title("PPO vs GRPO on GSM8K")
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

legend_handles = [
    mpatches.Patch(color=colors_peak["PPO"], label="PPO – Peak"),
    mpatches.Patch(color=colors_last10["PPO"], label="PPO – Last-10"),
    mpatches.Patch(color=colors_peak["GRPO"], label="GRPO – Peak"),
    mpatches.Patch(color=colors_last10["GRPO"], label="GRPO – Last-10"),
]
ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

# value labels
for i, mdl in enumerate(models):
    for algo in algos:
        vals = ppo_grpo[mdl][algo]
        for key, val in [("_peak", vals["peak"]), ("_last10", vals["last10"])]:
            off = offsets[f"{algo}{key}"]
            ax.text(
                x[i] + off,
                val + 1.5,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )

plt.tight_layout()
plt.savefig(OUTDIR + "ppo_vs_grpo_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ ppo_vs_grpo_comparison.png")

# ═══════════════════════════════════════════════════════════
# Figure 5: Sensitivity heatmap
# ═══════════════════════════════════════════════════════════
# Category assignment
CATEGORY_MAP = {
    "scale_gsm8k_qwen3-8b": ("qwen3-8b", "Scaling"),
    "scale_gsm8k_qwen3.5-4b": ("qwen3.5-4b", "Scaling"),
    "scale_gsm8k_qwen3.5-27b": ("qwen3.5-27b", "Scaling"),
    "scale_gsm8k_qwen3-32b": ("qwen3-32b", "Scaling"),
    "frontier_gsm8k_deepseek-v3.1": ("deepseek-v3.1", "Frontier"),
    "frontier_gsm8k_nemotron-120b": ("nemotron-120b", "Frontier"),
    "frontier_gsm8k_qwen3-235b": ("qwen3-235b-moe", "Frontier"),
    "moe_gsm8k_qwen3-30b-inst": ("qwen3-30b-moe-inst", "MoE"),
    "moe_gsm8k_qwen3-30b-moe": ("qwen3-30b-moe", "MoE"),
    "scale_gsm8k_llama-8b-inst": ("llama-8b-inst", "Scaling"),
    "cross_tool_llama-8b-inst": ("llama-8b-inst", "Cross-Task"),
    "cross_tool_qwen3-32b": ("qwen3-32b", "Cross-Task"),
}

categories = ["Scaling", "Frontier", "MoE", "Cross-Task"]

# Build model list (ordered by category then peak)
row_order = [
    "qwen3.5-4b",
    "qwen3-8b",
    "qwen3.5-27b",
    "qwen3-32b",
    "llama-8b-inst",
    "deepseek-v3.1",
    "nemotron-120b",
    "qwen3-235b-moe",
    "qwen3-30b-moe-inst",
    "qwen3-30b-moe",
]

# Build matrix: rows=models, cols=categories, value=peak accuracy (%)
# Default to NaN (no data)
heat = {m: {c: np.nan for c in categories} for m in row_order}

for exp_name, (ms, cat) in CATEGORY_MAP.items():
    # find experiment
    for exp in data:
        if exp["experiment"] == exp_name:
            pk = exp.get("peak", exp.get("peak_accuracy", 0)) or 0
            heat[ms][cat] = pk * 100
            break

# build matrix
mat = np.array([[heat[m][c] for c in categories] for m in row_order], dtype=float)

fig, ax = plt.subplots(figsize=(8, 6))
import matplotlib.cm as cm

masked = np.ma.masked_invalid(mat)
im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

# grid lines
for i in range(len(row_order) + 1):
    ax.axhline(i - 0.5, color="white", linewidth=1.5)
for j in range(len(categories) + 1):
    ax.axvline(j - 0.5, color="white", linewidth=1.5)

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=11)
ax.set_yticks(range(len(row_order)))
ax.set_yticklabels([pname(m) for m in row_order], fontsize=9.5)

# cell text
for i in range(len(row_order)):
    for j in range(len(categories)):
        val = mat[i, j]
        if not np.isnan(val):
            txt = f"{val:.0f}%"
            color = "white" if val < 30 or val > 80 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color, fontweight="bold")
        else:
            ax.text(j, i, "–", ha="center", va="center", fontsize=11, color="#aaaaaa")

cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cb.set_label("Peak Accuracy (%)", fontsize=10)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax.set_title("Model Performance by Experiment Category (GSM8K)", pad=14)
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
ax.grid(False)

# partial marker
partial_models_cats = set()
for exp_name, (ms, cat) in CATEGORY_MAP.items():
    for exp in data:
        if exp["experiment"] == exp_name and is_partial(exp):
            partial_models_cats.add((ms, cat))

for ms, cat in partial_models_cats:
    if ms in row_order and cat in categories:
        i = row_order.index(ms)
        j = categories.index(cat)
        ax.text(
            j + 0.38,
            i - 0.38,
            "*",
            ha="right",
            va="top",
            fontsize=10,
            color="#333333",
            fontweight="bold",
        )

ax.text(
    1.02, -0.04, "* partial run", transform=ax.transAxes, fontsize=8, color="#555555", ha="left"
)

plt.tight_layout()
plt.savefig(OUTDIR + "sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ sensitivity_heatmap.png")

print("\nAll figures saved to", OUTDIR)
