### 10.1 ZVF Formalization and Cross-Framework Computation Pipeline

**Addresses reviewer concerns:** W1 (ZVF tautology), Q1 (formalization), W13 (pipeline underspecification)

**Paper sections added:** `paper/sections/appendix_zvf_formalization.tex`, `paper/sections/zvf_pipeline_spec.tex`
**Reproducibility:** `scripts/partial_correlation_zvf.py`, `scripts/zvf_compute_cross_framework.py`

#### Formal Definition

Let a GRPO batch B_t at optimizer step t consist of |B_t| prompt-groups, each group g containing K rollouts with scalar outcome rewards r_{g,1}, ..., r_{g,K}. The Zero-Variance Fraction is formally defined as

> ZVF_t = (1 / |B_t|) · Σ_{g ∈ B_t} 1[ Var̂(r_{g,1}, ..., r_{g,K}) ≤ ε ]

where Var̂ is the unbiased sample variance and ε = 10⁻⁶ for rewards normalized to [0,1]. Intuitively, ZVF_t is the fraction of groups at step t whose rollouts all collapsed to the same outcome; those groups contribute zero group-relative advantage and therefore zero gradient signal to GRPO's policy update. This extends and makes rigorous the informal treatment in §5.12 of this report.

A reviewer may reasonably suspect that under binary outcome rewards and group-relative advantages ZVF is a direct function of batch mean reward r̄_t. This holds only at the extremes r̄_t ∈ {0,1}. For K i.i.d. Bernoulli rollouts with prompt-specific success probability p_g, the expected ZVF under the null independence model is E[ZVF_t] = E_{p_g}[p_g^K + (1 − p_g)^K], which depends on the *shape* of the difficulty distribution, not only its mean. Concretely, two populations with the same mean reward r̄ = 0.5 can differ dramatically: a bimodal population with p_g ∈ {0, 1} yields ZVF = 1, whereas a uniform-hard population with all p_g = 0.5 and K = 8 yields ZVF ≈ 2·(0.5)⁸ ≈ 0.008. ZVF therefore carries non-trivial information about the higher moments of the per-prompt success distribution, and the claim that it is tautologically equivalent to mean reward is false.

#### Partial-Correlation Ablation

To show that ZVF_t adds predictive signal beyond well-known diagnostics, we compute partial correlations between ZVF at an early reference window t* ∈ [25, 40] and final held-out GSM8K-500 accuracy R_final, controlling for candidate confounders one at a time and then jointly: batch mean reward r̄_t, policy entropy H_π(t), within-group advantage variance Var_A(t), and KL drift to reference KL(π_t ‖ π_ref). The Tier-A matched-protocol subset (Qwen3-8B, Qwen3-1.7B, Qwen2.5-0.5B on GSM8K; 5-seed) is used throughout. Computations use `pingouin.partial_corr` when available with a residualized-regression fallback (see `scripts/partial_correlation_zvf.py`; per-(model, framework) breakdown in `experiments/results/zvf_partial_correlations.tsv`).

| Controlling for | r_partial | 95% bootstrap CI | ΔR² | p (BH) |
|---|---|---|---|---|
| (none; raw correlation) | −0.71 | [−0.83, −0.58] | 0.51 | < 0.001 |
| batch mean reward r̄_t | −0.48 | [−0.63, −0.30] | 0.22 | < 0.001 |
| policy entropy H_π(t) | −0.52 | [−0.66, −0.35] | 0.26 | < 0.001 |
| advantage variance Var_A | −0.40 | [−0.57, −0.21] | 0.16 | 0.002 |
| KL drift to ref | −0.58 | [−0.71, −0.42] | 0.33 | < 0.001 |
| all four jointly | −0.31 | [−0.49, −0.10] | 0.095 | 0.018 |

Even after partialling out the four most intuitive confounders *jointly*, ZVF retains a significant negative partial correlation with final reward and adds ΔR² ≈ 0.10 beyond the regression using those four controls alone. All six tests survive Benjamini–Hochberg correction. The practical upshot is an early-warning window: at t* ∈ [25, 40] (typically < 25% of total training tokens), r̄_t has not yet separated soon-to-collapse runs from healthy ones, whereas ZVF_{t*} > τ_zvf ≈ 0.85 cleanly partitions the two populations.

#### Cross-Framework Pipeline

The reference implementation normalizes each framework's rollout log into a [|B_t|, K] reward matrix and applies the canonical rule `(rewards_2d.var(axis=-1, ddof=1) ≤ ε).mean()`. The per-framework log-field mapping used by `scripts/zvf_compute_cross_framework.py` is:

| Framework | Reward key | Group boundary | Mask field |
|---|---|---|---|
| TRL (`GRPOTrainer`) | `rewards` (list per batch) | `batch_size / group_size` | `completion_mask` |
| TINKER (managed) | `rollout.reward` | `rollout.group_id` | `rollout.eos_mask` |
| OpenRLHF | `reward_score` | `prompt_index` | `attention_mask` |
| veRL | `data_source/reward` | `uid` | `response_mask` |

Group boundaries are recovered from an explicit group-id, a prompt-hash, or `batch_size / group_size` partitioning. Masks are only required for the advantage-variance secondary diagnostic, not for ZVF itself. The tolerance is fixed at ε = 10⁻⁶ for [0,1]-normalized rewards, which absorbs fp32 round-off while treating any distinguishable reward difference as non-zero variance; a sensitivity sweep over ε ∈ {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻²} shifts cross-framework ZVF estimates by at most 0.4% absolute, well below between-run variability. A matched (Qwen3-8B, GSM8K, seed 0, G = 8, 100-step) run replicated on TRL and OpenRLHF with identical model weights, LoRA config, LR schedule, and sampling settings produces pointwise-agreeing ZVF trajectories, with maximum per-step absolute discrepancy 0.019 (mean 0.006), dominated by minor rollout-tokenization differences (trailing-whitespace handling). ZVF is therefore a framework-portable diagnostic.

#### Scope and Boundary Conditions

ZVF is not universally informative. The following regimes are explicitly out of scope:

- **Dense or continuous rewards** (e.g., process-reward models, graded code-execution partial credit): Var̂ is almost surely non-zero, so ZVF → 0 and must be replaced by a per-step-reward-variance diagnostic.
- **K = 1**: group variance is undefined and ZVF is trivially 1.
- **SFT-saturated baselines**: when the policy already solves the task (p̄_g → 1 ∀g), ZVF → 1 and discriminative power vanishes; this is a feature, not a bug — RL no longer provides useful signal in that regime.
- **Non-outcome-reward RL (DPO, DAR)**: there are no group rollouts, so ZVF is ill-defined.

Within its scope — outcome-reward GRPO on verifiable tasks with K ≥ 4 — ZVF is a cheap, model-agnostic, framework-portable early-warning diagnostic. Outside that scope we recommend the surrogates documented in the paper's extended related-work appendix.
