### 10.4 Evidence-Tier Partition and F1–F5 Survival Analysis

**Addresses reviewer concerns:** W4 (single-seed short-horizon Tinker/API runs inflate headline numbers), W5 (BH corrections aggregate single-seed rows with multi-seed rows, anti-conservative for the latter), Q5 (which of F1–F5 actually survive when restricted to TRL, ≥5 seeds, ≥100 steps?).

**Paper section added:** `paper/sections/statistical_rigor_addendum.tex` (§App. Statistical Rigor Addendum, label `app:stat-rigor-addendum`).
**Reproducibility:** `experiments/survival_analysis.py` → `experiments/results/survival_analysis.tsv` (deterministic, `MASTER_SEED = 20260506`, matches `experiments/compute_statistics.py`). Consistent with the main capstone's §5.15 (power / BH) and §5.16 (deterministic statistical rigor pass).

#### Evidence Tiers

For a (framework, model, task, algorithm) group $g$, let $s_g$ be the seed count and $\tau_g$ the minimum training horizon. Tiers are defined by Eq. (tiers) in the appendix:

- **Tier A (inferential-grade):** $s_g \ge 5$ **and** $\tau_g \ge 100$ steps. Supports Welch two-sample tests at the paper's minimum detectable effect size ($d_{\min} \approx 2.02$ for $n_1{=}n_2{=}5$, $\alpha{=}0.05$, $1-\beta{=}0.80$).
- **Tier B (supporting):** $3 \le s_g \le 4$ **and** $50 \le \tau_g < 100$ steps. CIs and point estimates stated, but no standalone BH claim.
- **Tier C (descriptive only):** everything else — *all* Tinker API runs, the partial Modal Qwen3-32B PPO run, all frontier-MoE case studies, and every single-seed framework-gap cell. Retained as descriptive case studies; excluded from the BH $p$-value family.

The previous BH table in the main paper (Tables `main_results_stats` / `ppo_grpo_stats`) mixed all three tiers into a single $k=38$ family. The restricted BH below uses only Tier-A/B tests.

#### F1–F5 Survival Table

Transcribed from `experiments/results/survival_analysis.tsv` (verified against the tex projection — results match):

| Finding | Claim (short) | Tier-A support | Tier-B support | Cohen's $d$ | Bootstrap 95% CI | $n$ runs | BH-adj $p$ | Conclusion |
|---------|---------------|----------------|----------------|-------------|------------------|----------|------------|------------|
| **F1** | ZVF diagnostic | no | no | — | — | 0 | — | insufficient Tier-A/B data (downgraded → Tier-C, descriptive) |
| **F2** | Instruct > Base trainability | no | no | — | — | 0 | — | insufficient Tier-A/B data (downgraded → Tier-C, descriptive) |
| **F3** | PPO/GRPO heterogeneity | **yes** | no | **+22.436** | [0.684, 0.761] | 40 | $6.72 \times 10^{-11}$ | **survives restricted BH** ($|d| \gg 0.3$, $p_{\text{BH}} \ll 0.05$) |
| **F4** | Framework gap (Qwen3-8B four-way) | no | no | — | — | 0 | — | insufficient Tier-A/B data (downgraded → Tier-C, descriptive) |
| **F5** | Frontier-MoE stability | no | no | — | — | 0 | — | insufficient Tier-A/B data (downgraded → Tier-C, descriptive) |

The headline picture is honest and conservative: **only F3 survives**. F3 is driven by the $n_1{=}n_2{=}5$ arithmetic seeds on TRL/SB3/CleanRL/Tianshou (40 runs total); the enormous effect size ($d \approx 22.4$) reflects the near-total PPO-vs-GRPO separation documented in §5.15. F1/F2/F4/F5 have no Tier-A or Tier-B groups at all in the current release (`n_runs_used = 0`) — the tex appendix uses symbols "✓ / ⚠ / ✓ / ⚠ / ✗" but the TSV makes clear the underlying verdict is uniformly "no Tier-A/B data" for the four non-surviving findings, not "partial support."

**Divergence note:** None. The tex appendix's Table `tab:survival-f1-f5` and the TSV produced by `experiments/survival_analysis.py` agree: F3 survives with $d{=}+22.44$, $p_{\text{BH}}{=}6.7\times 10^{-11}$; F1/F2/F4/F5 are all `insufficient Tier-A/B data`. The tex's narrative interpretation is slightly more charitable (e.g. "observed across Tinker runs" for F2) than the TSV's flat "no support," but the quantitative verdict is identical. This fragment reports the TSV's flat verdict.

#### Restricted BH Methodology

Let $\mathcal{F}_{AB} \subseteq \{1, \ldots, 38\}$ be the restricted family of Tier-A/B tests with $m' = |\mathcal{F}_{AB}|$. Applying Benjamini–Hochberg at FDR $q = 0.05$:

$$
p^{(i)} \;\le\; \frac{\operatorname{rank}(p^{(i)})}{m'} \cdot q, \qquad i \in \mathcal{F}_{AB},
$$

where ranks are taken within $\mathcal{F}_{AB}$ only. This is strictly less anti-conservative than the original paper-wide correction because Tier-C rows (which have $n_1{=}n_2{=}1$, undefined Welch $t$, and no within-group variance) previously inflated the BH denominator $m$. Tier-C results are reported descriptively in the main text with raw Welch $p$-values where they are algebraically defined, but they are **explicitly labelled "not corrected"** in the exported TSV and carry no significance claim.

#### Claims Downgraded Because They Relied on Tier-C

Per `paper/sections/statistical_rigor_addendum.tex` §downgrades, the following main-text claims are revised from inferential to descriptive:

- **F1 (ZVF as diagnostic).** Supported only by short-horizon Tinker API logs ($\tau < 100$, $s = 1$). Reported as a qualitative signature, not a tested effect.
- **F2 (Instruct > Base trainability).** Observed across Tinker runs but without paired $\ge 5$-seed $\ge 100$-step instruct/base comparisons in an open framework. Descriptive only.
- **F4 (Framework gap, Qwen3-8B).** The Tinker / TRL / veRL / OpenRLHF four-way comparison is $s_g = 1$ per cell (Tier-C). The 17× last-10-reward ratio between Tinker and TRL is reported as a descriptive ratio, not a tested effect.
- **F5 ("sustained ceiling" / frontier stability).** Every Tinker MoE run is single-seed short-horizon (20–30 steps). The "sustained ceiling" language is now a **descriptive observation scoped to the 20–30-step horizon**, not a mechanistic claim about frontier-model dynamics. See §10.7 for the full F5 scope clarification (frontier-MoE case-study framing + explicit horizon caveats).
- **Frontier training-reward rows in Table `tab:main_results`** (Qwen3-32B, Qwen3.5-27B, Nemotron-120B, Qwen3-235B-A22B, Qwen3-30B-A3B ×2, Kimi-K2 variants, GPT-OSS-120B, DeepSeek-V3.1): all Tier-C. The word "significant" is struck from their textual discussion; no BH-adjusted $p$-values attached.
- **PPO vs. GRPO on Qwen3-8B** (single-seed pair 0.344 / 0.350, Welch $p = 0.973$): preserved for reference in Table `tab:ppo_grpo_stats` but demoted to "descriptive" in the caption.
- **Frontier SI/PTD proxies:** per-run SI and PTD values for all Tinker MoE runs no longer count as observations in the paper-wide BH family.

The **negative held-out GSM8K result** (Qwen3-8B post-GRPO 83.3% vs. base 82.0%, $p = 0.26$) is unaffected — it is 5-seed TRL with a common eval harness (Tier-A). Its non-significance is, if anything, *strengthened* by restricting to the Tier-A/B family.

**Bottom line:** F3 is the one finding that clears the Q5 bar. F1, F2, F4, F5 are downgraded to descriptive case studies pending matched multi-seed ≥100-step Tier-A evidence in an open framework — the next-step programme flagged in the Conclusion.
