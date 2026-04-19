"""
Statistical Analysis Tooling for RL Experiments
=================================================
Implements rliable-based aggregate metrics, bootstrap confidence intervals,
and proper statistical testing following:

- Colas et al., "A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms" (2019)
  https://arxiv.org/abs/1904.06979
- Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" (2021)
  https://arxiv.org/abs/2108.13264
- Patterson et al., "Empirical Design in Reinforcement Learning" (2024)
  https://arxiv.org/abs/2304.01315

Usage:
    python utils/stats.py --results-dir results/ --output-dir paper/figures/
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_multi_seed_results(results_dir: str, experiment: str) -> Dict[int, List[float]]:
    """
    Load results from multiple seeds for a given experiment.

    Expected directory structure:
        results/<experiment>/seed_<N>/metrics.jsonl

    Returns:
        Dict mapping seed -> list of metric values over training steps.
    """
    seed_results = {}
    pattern = os.path.join(results_dir, experiment, "seed_*", "*.jsonl")
    for filepath in sorted(glob.glob(pattern)):
        seed_dir = os.path.basename(os.path.dirname(filepath))
        seed = int(seed_dir.replace("seed_", ""))
        metrics = []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                metrics.append(data)
        seed_results[seed] = metrics
    return seed_results


def compute_bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        scores: Array of scores (one per seed/run).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        rng: NumPy random generator for reproducibility.

    Returns:
        (mean, lower_ci, upper_ci)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(scores)
    bootstrap_means = np.array(
        [np.mean(rng.choice(scores, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, 100 * alpha)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha))
    mean = np.mean(scores)

    return mean, lower, upper


def welch_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Welch's t-test for comparing two algorithms.
    Recommended over Student's t-test when variances may differ.

    Reference: Colas et al. (2019), Section 4.1
    """
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    effect_size = (np.mean(scores_a) - np.mean(scores_b)) / np.sqrt(
        (np.var(scores_a) + np.var(scores_b)) / 2
    )

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "effect_size_cohens_d": float(effect_size),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "std_a": float(np.std(scores_a, ddof=1)),
        "std_b": float(np.std(scores_b, ddof=1)),
        "n_a": len(scores_a),
        "n_b": len(scores_b),
    }


def mann_whitney_u(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).
    Use when distributions may be non-normal.

    Reference: Colas et al. (2019), Section 4.2
    """
    from scipy import stats

    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")

    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "significant_at_005": bool(p_value < 0.05),
        "median_a": float(np.median(scores_a)),
        "median_b": float(np.median(scores_b)),
    }


def plot_learning_curves_with_ci(
    results: Dict[str, Dict[int, List[dict]]],
    metric_key: str = "reward/mean",
    output_path: str = "learning_curves.pdf",
    title: str = "Learning Curves with 95% Confidence Intervals",
):
    """
    Plot learning curves with shaded confidence bands (±1 SE).

    Args:
        results: Dict[algorithm_name -> Dict[seed -> List[step_metrics]]]
        metric_key: Which metric to plot
        output_path: Where to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = sns.color_palette("colorblind", n_colors=len(results))

    for idx, (algo_name, seed_data) in enumerate(results.items()):
        # Align all seeds to same number of steps
        all_curves = []
        for seed, metrics_list in seed_data.items():
            curve = [m.get(metric_key, 0) for m in metrics_list]
            all_curves.append(curve)

        # Truncate to shortest run
        min_len = min(len(c) for c in all_curves)
        all_curves = np.array([c[:min_len] for c in all_curves])

        mean = np.mean(all_curves, axis=0)
        se = np.std(all_curves, axis=0, ddof=1) / np.sqrt(len(all_curves))
        steps = np.arange(1, min_len + 1)

        ax.plot(steps, mean, label=algo_name, color=colors[idx], linewidth=2)
        ax.fill_between(
            steps,
            mean - 1.96 * se,
            mean + 1.96 * se,
            alpha=0.2,
            color=colors[idx],
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(metric_key, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to {output_path}")


def generate_results_table(
    results: Dict[str, np.ndarray],
    output_path: str = "results_table.tex",
    metric_name: str = "Accuracy",
):
    """
    Generate a LaTeX table with mean ± SE and bootstrap CIs.

    Args:
        results: Dict[algorithm_name -> array of final scores across seeds]
        output_path: Where to save the LaTeX table
        metric_name: Column header for the metric
    """
    rows = []
    for algo_name, scores in results.items():
        mean, ci_lower, ci_upper = compute_bootstrap_ci(scores)
        se = np.std(scores, ddof=1) / np.sqrt(len(scores))
        rows.append(
            {
                "Algorithm": algo_name,
                f"{metric_name} (mean ± SE)": f"{mean:.3f} ± {se:.3f}",
                "95% CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                "Seeds": len(scores),
            }
        )

    df = pd.DataFrame(rows)

    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved results table to {output_path}")

    # Also save as CSV
    csv_path = output_path.replace(".tex", ".csv")
    df.to_csv(csv_path, index=False)

    return df


def try_rliable_analysis(results: Dict[str, np.ndarray], output_dir: str):
    """
    Run rliable aggregate metrics if the library is available.

    Reference: Agarwal et al. (2021)
    https://arxiv.org/abs/2108.13264
    """
    try:
        from rliable import library as rly
        from rliable import metrics as rly_metrics

        # ``plot_utils`` is imported to verify the full rliable install is
        # present (the caller later builds rliable plots via helper scripts);
        # we assign ``_`` to signal the availability check to ruff/linters.
        from rliable import plot_utils as _  # noqa: F401

        print("Running rliable analysis...")

        # Prepare score dictionaries
        score_dict = {}
        for algo, scores in results.items():
            # rliable expects (n_runs, n_tasks) array
            score_dict[algo] = scores.reshape(-1, 1) if scores.ndim == 1 else scores

        # Compute aggregate metrics with CIs
        aggregate_func = lambda x: np.array(
            [
                rly_metrics.aggregate_median(x),
                rly_metrics.aggregate_iqm(x),
                rly_metrics.aggregate_mean(x),
                rly_metrics.aggregate_optimality_gap(x),
            ]
        )

        aggregate_scores, aggregate_cis = rly.get_interval_estimates(
            score_dict, aggregate_func, reps=50000
        )

        # Save rliable results
        rliable_results = {
            "aggregate_scores": {k: v.tolist() for k, v in aggregate_scores.items()},
            "aggregate_cis": {k: v.tolist() for k, v in aggregate_cis.items()},
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "rliable_results.json"), "w") as f:
            json.dump(rliable_results, f, indent=2)

        print(f"rliable results saved to {output_dir}/rliable_results.json")

    except ImportError:
        print("rliable not installed. Install with: pip install rliable")
        print("Falling back to bootstrap CI analysis.")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for RL experiments")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument(
        "--experiment", type=str, default=None, help="Specific experiment to analyze"
    )
    parser.add_argument("--output-dir", type=str, default="paper/figures/")
    parser.add_argument("--format", type=str, choices=["latex", "csv", "both"], default="both")
    parser.add_argument("--rliable", action="store_true", help="Run rliable aggregate analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover experiments
    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = [
            d
            for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
        ]

    print(f"Found experiments: {experiments}")

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {exp}")
        print(f"{'=' * 60}")

        seed_results = load_multi_seed_results(args.results_dir, exp)
        if not seed_results:
            print(f"  No multi-seed results found for {exp}")
            continue

        print(f"  Found {len(seed_results)} seeds: {list(seed_results.keys())}")

        # Extract final scores for each seed
        final_scores = []
        for seed, metrics_list in seed_results.items():
            if metrics_list:
                last_metric = metrics_list[-1]
                score = last_metric.get(
                    "reward/mean",
                    last_metric.get("accuracy", last_metric.get("eval/percent_correct", 0)),
                )
                final_scores.append(score)

        if final_scores:
            scores_arr = np.array(final_scores)
            mean, ci_lower, ci_upper = compute_bootstrap_ci(scores_arr)
            se = np.std(scores_arr, ddof=1) / np.sqrt(len(scores_arr))
            print(
                f"  Final score: {mean:.4f} ± {se:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
            )

    print("\nStatistical analysis complete.")


# ----------------------------------------------------------------------------
# Compatibility aliases
# ----------------------------------------------------------------------------
# Some downstream tooling / CI imports the bootstrap CI under a shorter name.
bootstrap_ci = compute_bootstrap_ci


def compute_iqm(scores: np.ndarray, tau: float = 0.25) -> float:
    """Compute the Interquartile Mean (IQM) of a score array.

    Follows Agarwal et al. (2021), "Deep RL at the Edge of the Statistical
    Precipice". Drops the top and bottom `tau` fraction of values and returns
    the mean of the middle (1 - 2*tau) fraction. Default tau=0.25 gives the
    canonical IQM over the central 50% of scores.
    """
    arr = np.asarray(scores, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    lo = np.quantile(arr, tau)
    hi = np.quantile(arr, 1 - tau)
    mid = arr[(arr >= lo) & (arr <= hi)]
    if mid.size == 0:
        return float(np.mean(arr))
    return float(np.mean(mid))


if __name__ == "__main__":
    main()
