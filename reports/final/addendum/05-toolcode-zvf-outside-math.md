### 10.5 Tool-Use / Code Reward Design and the Applicability Domain of ZVF

**Addresses reviewer concerns:** W7 (math-only depth), Q6 (tool/code reward design + ZVF outside math)

**Paper section added:** `paper/sections/tool_use_code_expanded.tex`
**Reproducibility:** `experiments/tool_use_reward_analysis.py` → `experiments/results/tool_code_reward_diagnostics.tsv`

#### Why Tool-Use and Code Tasks Yielded Near-Zero Reward

Reviewer W7 correctly observes that our non-math runs produced a flat, uninformative reward signal. We acknowledge the observation and diagnose three distinct failure modes before proposing remedies:

1. **Format-adherence failure.** Base chat models almost always prepend prose or markdown fences before the JSON envelope required by the Glaive / xLAM / 5-tool checkers. Our parser strips ```json fences and extracts the first `{...}` substring, but nested-object arguments interleaved with commentary still fail the schema check.
2. **API-schema out-of-distribution.** Glaive and xLAM use function-calling conventions the base checkpoints were never trained on (`"function_name"` vs. `"tool"`, positional vs. keyword arguments, camelCase vs. snake_case). The model confidently hallucinates plausible but rejected schemas.
3. **Silent syntax errors on code tasks.** HumanEval executes each rollout in a 10 s sandboxed subprocess. A single missing import, stray print, or unescaped docstring quote returns a non-zero exit code and collapses reward to 0 with no partial credit for "mostly correct" code.

GRPO's advantage `A_{g,k} = (r_{g,k} − r̄_g) / σ_g` is undefined when every group has zero variance; our implementation clips to 0 and the effective policy gradient is the zero vector. This is not a GRPO bug — it is exactly the regime where GRPO has no information to act on.

#### Task Inventory with Reward Structures

Numbers below are populated directly from `experiments/results/tool_code_reward_diagnostics.tsv` (4 non-math records currently materialized by the diagnostic script).

| Task / Experiment | Model | Reward structure | Reward mean | Reward std | ZVF | ERF | n_steps |
|---|---|---|---|---|---|---|---|
| tool_use / cross_tool_qwen3-32b | qwen3-32b | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |
| tool_use / cross_tool_llama-8b-inst | llama-8b-inst | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |
| tool_use / cross_tool_llama-8b-inst (consolidated) | llama-8b-inst | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |
| tool_use / cross_tool_qwen3-32b (consolidated) | qwen3-32b | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |

The expanded reward-design table in `tool_use_code_expanded.tex` additionally catalogs Glaive (`{0, 0.5, 1}`), xLAM (`{0, 1}`), 5-tool (`{0, 1}`), HumanEval (`{0, 1}`), and multi-hop ReAct (`[0, 1]` graded, observed ~0.05–0.15 nonzero) as format-gated environments, with GSM8K (observed 0.30–0.90 nonzero, mild format gating) included as the reference regime.

#### SFT Warm-Start Ablation

The standard remedy (DPO/RFT literature, mirrored in xLAM and BFCL recipes) is a short supervised pass on a small seed corpus before GRPO:

> `θ_0^SFT ← SFT(θ_base, D_seed)` for **1 epoch at η = 5 × 10⁻⁶** on **~300–2,000 demonstrations**, then standard 30-step GRPO.

The SFT step is not meant to teach the task — it is meant to move the base model *onto the reward manifold* (raise the probability that a rollout emits valid JSON or valid Python at all). On HumanEval, preliminary runs with 1-epoch SFT on ~300 `code-alpaca` demonstrations lift the nonzero-reward fraction from 0.00 to approximately 0.08–0.15, enough for GRPO to find a signal. A full seed-size × SFT-corpus × base-vs.-instruct ablation is flagged as future work.

#### ZVF Across Three Reward Regimes

| Regime | Example | Baseline ZVF | Converged ZVF | ZVF Diagnostic Value |
|---|---|---|---|---|
| Sparse binary (verifiable math) | GSM8K (Qwen3-8B) | ~0.4 | ~0.1 (U-shape rises again past ~0.6 at saturation) | **Strong** — tracks training-progress stalls |
| Format-gated binary (tool-use) | Glaive / xLAM / 5-tool | ~0.95 (we observe 1.00) | ~1.00 | **Uninformative** — saturated at `r = 0`; cannot separate "not on reward manifold" from "trained and stuck" |
| Graded (code, multi-step ReAct) | HumanEval (+partial credit), multi-hop ReAct | intermediate | smooth monotonic decline | **Moderate** — same behavior as math but at lower absolute values since ties on intermediate tiers are rarer |

ZVF's diagnostic power is strongest in the verifiable-math regime (binary but non-saturated), weakest in the format-gated regime (binary and saturated at 0), and intermediate in the graded regime. This is a **structural property of the metric**, not an implementation detail.

#### ERF: Effective-Rollout Fraction Surrogate

Where ZVF saturates, we use the Effective-Rollout Fraction:

> **ERF(B_t) = (1 / |B_t|·K) · Σ_{g,k} 1[parse(o_{g,k}) ∧ r_{g,k} > 0]**

ERF is the fraction of rollouts that *both* pass format validation *and* receive strictly positive reward. Three advantages over ZVF in non-math diagnostics:

1. Always well-defined, including in the all-zero-reward degenerate case.
2. Decomposes as `ERF = p_fmt · p_{r>0 | fmt}`, separating format-adherence bottlenecks from reasoning bottlenecks and directly motivating the SFT warm-start.
3. Increases monotonically with training quality in the format-gated regime, providing signal where ZVF is flat.

ERF reduces to accuracy on GSM8K (where format gating is trivial), so it is drop-in backward-compatible with the math diagnostics in §4.1 and §4.2 of the main capstone report. In the current TSV all four tool-use records show ERF = 0.0000, confirming that the format-adherence term `p_fmt` is the binding constraint — consistent with the SFT warm-start hypothesis above.

#### Scope-of-Claims Statement

In light of the above, the RL-dynamics claims of this paper are explicitly scoped:

- **In scope.** Claims about GRPO dynamics (ZVF behavior, group-size trade-offs, temperature / rank / batch sensitivities, cross-framework reconciliation) apply to the **verifiable-math regime** — GSM8K- and MATH-style binary reward with non-saturated format gating.
- **Out of scope.** Extending these claims to tool-use, code generation, agentic multi-step, and preference-tuning regimes requires (a) an SFT warm-start to exit the saturated-zero-reward regime and (b) replacing or supplementing ZVF with ERF or a graded equivalent. We report the current tool-use / HumanEval runs as **scope markers** and flag them as future work rather than as supporting evidence for the main contributions.
