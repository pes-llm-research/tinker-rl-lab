### 10.7 F5 Scope Clarification and Regenerated Figures

**Addresses reviewer concerns:** W10 (F5 brief runs), W12 (placeholder figures)

**Paper sections added:** `paper/sections/frontier_scope_clarification.tex`, `paper/sections/figures_regeneration_note.tex`
**Reproducibility:** `scripts/regenerate_missing_figures.py`

---

#### 10.7.1 F5 Reframed from Mechanistic to Descriptive

Reviewer feedback (W10) correctly flagged that the earlier phrasing of F5 —
"sustained ceiling / stability at frontier scale" — over-generalised what the
underlying Tinker-backed runs can support. The frontier evidence consists of
short-horizon (20–30 step), mostly single-seed, occasionally interrupted runs
on API-managed models at N ≥ 70B parameters. The revised F5' statement in the
paper is:

> *At the frontier scales we tested (N ≥ 70B) over 20–30 steps of Tinker-managed
> GRPO training on GSM8K, we observe that reward trajectories for a subset of
> reasoning-tuned checkpoints remain bounded within [baseline, baseline + ε]
> without the oscillation/collapse patterns visible at 0.6B–8B for some base
> checkpoints; we do **not** claim this generalises beyond the evaluated 20–30
> step horizon, to additional seeds, to other initialisations, or to tasks
> beyond GSM8K training reward.*

Any residual monotonic-in-N reading is **explicitly retracted**. Our own data
contains direct counterexamples against a monotonic stability claim:
Nemotron-120B (87.5% peak → 16.2% last-10 regression), Qwen3-32B (31.2% peak,
run interrupted), and Kimi-K2.5 (100% → 62.5% drop). These are reported as
non-monotonic excursions outside the exponential saturation regime of
Section 5.8 (Frontier Model Scaling Laws), not as evidence for F5'.

##### Evidence-Strength Tier Table for F5

| Tier                   | Seeds       | Steps    | Scope in this paper                         | Usage                |
| ---------------------- | ----------- | -------- | ------------------------------------------- | -------------------- |
| Strong evidence        | ≥ 5         | ≥ 100    | 0.6B–8B GSM8K GRPO                          | Quantitative claim   |
| Supportive evidence    | ≥ 3         | ≥ 50     | 14B–32B GSM8K GRPO (partial)                | Trend statement      |
| Descriptive obs. (F5') | 1 (typical) | 15–30    | 70B–671B Tinker API runs                    | Illustrative, not law|
| Interrupted / partial  | 1           | < 20     | Subset of frontier runs (marked † in tables)| Footnote only        |

All frontier runs used to illustrate F5' fall in the third row and are
**never** aggregated with the multi-seed small-scale runs when stating any
scaling-law claim.

##### What Would Upgrade F5 to a Robust Claim

A defensible replication requires **5 seeds × 200 steps × 3 frontier sizes
≈ 30 runs** at (70B, 120B, 235B), with matched rollout budgets and held-out
evaluation (not just training reward). Aggregate cost: roughly 60× the budget
of the current single-seed short runs, i.e. on the order of 10⁴–10⁵ USD of
Tinker-managed compute per frontier size. This sits outside the present
release's budget envelope; the released scripts and configs are structured so
such a replication can reuse the same evaluation harness unchanged.

---

#### 10.7.2 Regenerated Figures

Reviewer W12 flagged that three figure PDFs had shipped as placeholder boxes
in an earlier draft. All three have been regenerated as real rendered PDF+PNG
pairs via a single entrypoint, `scripts/regenerate_missing_figures.py`, which
consumes canonical data files (or falls back deterministically to the numbers
already quoted in the paper text when a source file is missing). The script
depends only on `numpy` and `matplotlib` — no `scipy` — so it runs in the
minimal environment used for artefact review.

| Figure                | Source Data                                                                     | Size                       |
| --------------------- | ------------------------------------------------------------------------------- | -------------------------- |
| `performance_profiles`| `experiments/results/arithmetic_metrics.jsonl` + 4 simulated library distributions | 25.6 KB PDF / 70.8 KB PNG |
| `wave6_sensitivity`   | `experiments/tinker-runs/results/wave6_ablations.json` (real Wave-6 sweeps)     | 35.5 KB PDF / 197 KB PNG   |
| `old_trl_seeds`       | Canonical summary: mean 73.4%, CV 0.096, t = 7.44, p < 0.001 (5-seed TRL GRPO)  | 29.7 KB PDF / 86.8 KB PNG  |

The paper's `main.tex` wraps each `\includegraphics` in
`\IfFileExists{...}{real}{placeholder}`, so the document compiles whether the
regenerated figures are present or not. After running
`python3 scripts/regenerate_missing_figures.py`, all six files are written to
the exact paths `main.tex` expects and the real-figure branch fires for all
three, eliminating the placeholder boxes without any change to the LaTeX
source.
