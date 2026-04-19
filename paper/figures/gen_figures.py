"""
Generate publication-quality figures:
1. Performance profiles (Dolan-Moré style)
2. Hyperparameter sensitivity heatmap
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

OUT_DIR = "/home/user/workspace/tinker-rl-lab/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    }
)

# Design palette (from skill)
COLORS = {
    "GRPO-DeepSeek-V3.1": "#20808D",  # Teal
    "GRPO-Qwen3-8B": "#A84B2F",  # Terra/rust
    "PPO-Qwen3-8B": "#944454",  # Mauve
    "PPO-Llama-8B": "#1B474D",  # Dark teal
}

# ─── Reward traces ─────────────────────────────────────────────────────────────
traces = {
    "GRPO-DeepSeek-V3.1": np.array(
        [
            0.875,
            0.875,
            1.0,
            0.75,
            0.75,
            0.75,
            0.625,
            0.875,
            0.875,
            1.0,
            0.75,
            1.0,
            0.875,
            0.875,
            0.75,
            1.0,
            0.875,
            0.75,
            0.875,
            0.875,
        ]
    ),
    "GRPO-Qwen3-8B": np.array(
        [
            0.375,
            0.0625,
            0.5625,
            0.1875,
            0.375,
            0.3125,
            0.125,
            0.125,
            0.1875,
            0.125,
            0.25,
            0.4375,
            0.625,
            0.1875,
            0.3125,
            0.5,
            0.375,
            0.375,
            0.4375,
            0.5,
            0.3125,
            0.125,
            0.25,
            0.4375,
            0.5,
            0.3125,
            0.4375,
            0.375,
            0.1875,
            0.25,
        ]
    ),
    "PPO-Qwen3-8B": np.array(
        [
            0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            0.5,
            0.0,
            0.0,
            0.25,
            0.5,
            0.5,
            0.0,
            0.5,
            0.25,
            0.0,
            0.0,
            0.0,
            0.25,
            0.25,
            0.0,
            0.5,
            0.5,
            0.5,
            0.0,
            0.75,
            0.25,
            0.0,
            0.25,
            0.75,
            0.0,
        ]
    ),
    "PPO-Llama-8B": np.array(
        [
            1.0,
            1.0,
            1.0,
            0.75,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.75,
            1.0,
            1.0,
            1.0,
            1.0,
            0.75,
            1.0,
            1.0,
            1.0,
            1.0,
            0.75,
            1.0,
        ]
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Performance profiles (Dolan-Moré style)
# ══════════════════════════════════════════════════════════════════════════════
# In Dolan-Moré profiles for RL/ML:
# - Each "problem" = one step in the reward trace
# - For each step i, define r_i(method) = best_reward_i / method_reward_i
#   BUT since higher reward is better (not lower cost), we invert:
#   ratio = best_at_step_i / method_reward_i  → values >= 1
#   where best_at_step_i = max over all methods at that step
# - rho(tau) = fraction of steps where ratio <= tau
# We use the union of all steps across methods.

# Align: pad shorter traces to same length as longest by NaN
n_methods = len(traces)
method_names = list(traces.keys())

# Build matrix: rows = steps, cols = methods
# Use per-step scores for each method; NaN where trace shorter
max_len = max(len(t) for t in traces.values())

score_matrix = np.full((max_len, n_methods), np.nan)
for j, name in enumerate(method_names):
    t = traces[name]
    score_matrix[: len(t), j] = t

# For each step, compute best (max) reward across methods (ignoring NaN)
best_per_step = np.nanmax(score_matrix, axis=1)  # shape (max_len,)

# For each method, compute ratio = best_per_step / method_score_at_step
# Where method score is 0, we set ratio to infinity (method solved nothing)
# Ratio < 1 not possible (best is max); ratio == 1 means method is best at this step
eps = 1e-9
ratios = {}
for j, name in enumerate(method_names):
    method_scores = score_matrix[:, j]
    valid = ~np.isnan(method_scores)
    r = np.full(max_len, np.inf)
    # where method score > 0: ratio = best / method_score
    pos = valid & (method_scores > eps)
    r[pos] = best_per_step[pos] / method_scores[pos]
    # where method score == 0 and best > 0: ratio = inf (cannot solve)
    # where both are 0: treat as solved (ratio = 1)
    zero_both = valid & (method_scores <= eps) & (best_per_step <= eps)
    r[zero_both] = 1.0
    # only count valid steps
    ratios[name] = r[valid]

# Build profiles
tau_vals = np.linspace(1.0, 4.0, 500)

fig, ax = plt.subplots(figsize=(6.5, 4.5))

line_styles = {
    "GRPO-DeepSeek-V3.1": "-",
    "GRPO-Qwen3-8B": "--",
    "PPO-Qwen3-8B": "-.",
    "PPO-Llama-8B": ":",
}
line_widths = {
    "GRPO-DeepSeek-V3.1": 2.2,
    "GRPO-Qwen3-8B": 2.0,
    "PPO-Qwen3-8B": 2.0,
    "PPO-Llama-8B": 2.2,
}

for name in method_names:
    r = ratios[name]
    n_problems = len(r)
    rho = np.array([np.mean(r <= tau) for tau in tau_vals])
    ax.step(
        tau_vals,
        rho,
        where="post",
        color=COLORS[name],
        linestyle=line_styles[name],
        linewidth=line_widths[name],
        label=name,
        alpha=0.92,
    )

ax.set_xlabel(r"Performance ratio $\tau$", fontsize=13)
ax.set_ylabel(r"Fraction of steps solved within $\tau$", fontsize=13)
ax.set_xlim(1.0, 4.0)
ax.set_ylim(-0.02, 1.05)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Legend outside top-right
legend = ax.legend(
    loc="lower right",
    framealpha=0.9,
    edgecolor="#D4D1CA",
    fancybox=False,
)

ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT_DIR, "performance_profiles.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "performance_profiles.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved performance_profiles.png + .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Sensitivity heatmap — model × metric
# ══════════════════════════════════════════════════════════════════════════════
# Models (columns): Qwen2.5-0.5B (TRL GRPO), GRPO-Qwen3-8B, GRPO-DeepSeek-V3.1, PPO-Qwen3-8B, PPO-Llama-8B
# Metrics (rows): Peak Reward, Last-10 Avg, Steps to >50%, Volatility


# ── Helper functions ──────────────────────────────────────────────────────────
def peak_reward(trace):
    return float(np.max(trace))


def last10_avg(trace):
    return float(np.mean(trace[-10:]))


def steps_to_50(trace):
    """Steps (1-indexed) until first reward > 0.5. None if never."""
    for i, r in enumerate(trace):
        if r > 0.5:
            return i + 1
    return len(trace)  # never crossed; return total steps


def volatility(trace):
    return float(np.std(trace))


# ── Per-model stats ───────────────────────────────────────────────────────────
# Qwen2.5-0.5B (TRL GRPO, 5 seeds): only mean stats available
# We'll synthesise a plausible trace using the 5 seed accuracies as "steps"
trl_accuracies = np.array([0.735, 0.81, 0.62, 0.74, 0.765])
trl_mean = 0.734
trl_std = 0.065
# peak = max of seeds
trl_peak = float(np.max(trl_accuracies))
# last10_avg: approximate as mean (we only have 5 points)
trl_last10 = trl_mean
# steps_to_50: all seeds > 0.5, so it's step 1
trl_steps50 = 1
# volatility = std across seeds
trl_vol = trl_std

models = [
    "Qwen2.5-0.5B\n(TRL GRPO)",
    "GRPO\nQwen3-8B",
    "GRPO\nDeepSeek-V3.1",
    "PPO\nQwen3-8B",
    "PPO\nLlama-8B",
]

# Compute raw values
raw = {
    "Peak Reward": [
        trl_peak,
        peak_reward(traces["GRPO-Qwen3-8B"]),
        peak_reward(traces["GRPO-DeepSeek-V3.1"]),
        peak_reward(traces["PPO-Qwen3-8B"]),
        peak_reward(traces["PPO-Llama-8B"]),
    ],
    "Last-10 Avg": [
        trl_last10,
        last10_avg(traces["GRPO-Qwen3-8B"]),
        last10_avg(traces["GRPO-DeepSeek-V3.1"]),
        last10_avg(traces["PPO-Qwen3-8B"]),
        last10_avg(traces["PPO-Llama-8B"]),
    ],
    "Steps to >50%": [
        trl_steps50,
        steps_to_50(traces["GRPO-Qwen3-8B"]),
        steps_to_50(traces["GRPO-DeepSeek-V3.1"]),
        steps_to_50(traces["PPO-Qwen3-8B"]),
        steps_to_50(traces["PPO-Llama-8B"]),
    ],
    "Volatility\n(std)": [
        trl_vol,
        volatility(traces["GRPO-Qwen3-8B"]),
        volatility(traces["GRPO-DeepSeek-V3.1"]),
        volatility(traces["PPO-Qwen3-8B"]),
        volatility(traces["PPO-Llama-8B"]),
    ],
}

metric_names = list(raw.keys())
n_models = len(models)
n_metrics = len(metric_names)

# Build raw matrix (metrics × models)
raw_matrix = np.array([raw[m] for m in metric_names], dtype=float)

print("\nRaw values:")
for i, m in enumerate(metric_names):
    vals = [f"{v:.3f}" for v in raw_matrix[i]]
    print(f"  {m:20s}: {vals}")

# ── Normalize per row for coloring ───────────────────────────────────────────
# For "Steps to >50%": lower is better → invert direction for coloring
# For all others: higher is better
norm_matrix = np.zeros_like(raw_matrix)
is_lower_better = [False, False, True, False]  # Steps to >50% is lower-better

for i in range(n_metrics):
    row = raw_matrix[i].copy()
    rmin, rmax = row.min(), row.max()
    if rmax - rmin < 1e-9:
        norm_matrix[i] = 0.5 * np.ones(n_models)
    else:
        norm_row = (row - rmin) / (rmax - rmin)
        if is_lower_better[i]:
            norm_row = 1.0 - norm_row  # invert so "good" = high (green)
        norm_matrix[i] = norm_row


# ── Format annotation strings ────────────────────────────────────────────────
def fmt_val(metric_name, val):
    if "Steps" in metric_name:
        return f"{int(val)}"
    elif "Volatility" in metric_name:
        return f"{val:.3f}"
    else:
        return f"{val:.3f}"


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.2))

cmap = plt.get_cmap("RdYlGn")
im = ax.imshow(norm_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

# Cell annotations
for i in range(n_metrics):
    for j in range(n_models):
        raw_val = raw_matrix[i, j]
        norm_val = norm_matrix[i, j]
        text = fmt_val(metric_names[i], raw_val)
        # Adaptive text color based on cell luminance
        rgba = cmap(norm_val)
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "#28251D" if lum > 0.5 else "white"
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="medium",
            color=text_color,
        )

# Axes
ax.set_xticks(range(n_models))
ax.set_xticklabels(models, fontsize=10)
ax.set_yticks(range(n_metrics))
ax.set_yticks(range(n_metrics))

# Clean metric labels (remove embedded newlines for display)
clean_metric_names = [m.replace("\n", " ") for m in metric_names]
ax.set_yticklabels(clean_metric_names, fontsize=11)

ax.set_xlabel("Model / Training Method", fontsize=13, labelpad=8)
ax.set_ylabel("Metric", fontsize=13, labelpad=8)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Relative performance\n(row-normalized)", fontsize=10)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["Worst", "", "Mid", "", "Best"], fontsize=9)

# Grid lines between cells
ax.set_xticks(np.arange(-0.5, n_models, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_metrics, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(which="major", bottom=False, left=False)

# Remove outer spines
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT_DIR, "sensitivity_heatmap.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "sensitivity_heatmap.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved sensitivity_heatmap.png + .pdf")
