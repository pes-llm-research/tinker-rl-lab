# Submission-quality figures (v2)

These figures are generated deterministically from
`experiments/master_results.json` by
[`scripts/regenerate_figures.py`](../../../scripts/regenerate_figures.py).

Every figure is rendered at **300 dpi PNG** and as a **vector PDF** with the
following design rules:

- **Palette:** Okabe–Ito colorblind-safe (8 colors). Methods have a stable
  color assignment across all figures:
  - GRPO / Tinker GRPO → blue (`#0072B2`)
  - PPO-REINFORCE / Modal PPO → vermillion (`#D55E00`)
  - TRL GRPO → sky blue
  - Legacy PPO frameworks (SB3 / CleanRL / Tianshou) → orange / purple / green
  - Team members → bluish-green
- **Typography:** `font.family = "serif"` with a preferred stack of
  Times New Roman → STIXGeneral → Liberation Serif → DejaVu Serif, matched to
  the NeurIPS body text. All mathtext uses the `stix` fontset so inline math
  renders in the same face.
- **Legends:** Placed **outside** the axes whenever more than two series are
  present, using `bbox_to_anchor` or `loc="outside lower center"` with
  `constrained_layout` so no text overlaps plot data.
- **Layout:** `constrained_layout=True` on every figure; fonts embedded via
  `pdf.fonttype = 42` so reviewers can edit the PDF if needed.

## Figure inventory

| # | Stem                     | Description                                                              |
|---|--------------------------|--------------------------------------------------------------------------|
| 1 | `learning_curves`        | GRPO / PPO-REINFORCE reward curves for flagship GSM8K & tool-use runs.   |
| 2 | `comparison_bars`        | Mean ± std final reward by training family, with per-run scatter points. |
| 3 | `scaling`                | Parameter count (log x) vs last-10 GSM8K reward, with dense/MoE markers. |
| 4 | `ppo_vs_grpo`            | Paired GSM8K curves + peak/last-10 bars for Qwen3-8B and Llama-3.1-8B.   |
| 5 | `sensitivity_heatmap`    | Model × task reward grid (max over runs) with diverging colormap.        |
| 6 | `kl_proxy`               | ZVF fraction + surrogate loss per step across six flagship runs.         |
| 7 | `group_size_ablation`    | GRPO group-size sweep G ∈ {2, 4, 8, 16, 32} on Qwen3-8B / GSM8K.         |
| 8 | `framework_comparison`   | Horizontal bar comparison of Tinker GRPO vs legacy RL frameworks.        |

## Reproducing

```bash
python scripts/regenerate_figures.py
```

The script is idempotent and deterministic (`np.random.seed(7)` is fixed for
the small scatter jitter). It reads a single input file
(`experiments/master_results.json`) and writes 16 artifacts
(`*.png` + `*.pdf`) into this directory.

## Data provenance

Each figure pulls only from `experiments/master_results.json`, which is the
consolidated source assembled from:

- `experiments/all_results_consolidated.json`
- `experiments/tinker-runs/results/*`
- `experiments/modal/results/modal_parallel_results.json`
- `experiments/results/modal_results_all.json`

The legacy TRL GRPO baseline summary is read from the top-level
`trl_grpo_baseline_summary` block of the same JSON.
