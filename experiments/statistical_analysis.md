# Statistical Rigor Pass — Tinker RL Lab (NeurIPS 2026)

> **Master seed:** `20260506` · **Bootstrap:** B = 10,000 (percentile) · **CI:** 95 % · **α:** 0.05

Every comparison in the paper is reported with (i) 95 % bootstrap CI on the effect, (ii) Cohen's $d$ with a 95 % analytical CI, (iii) a raw p-value, and (iv) a Bonferroni-corrected p-value across the paper-wide family of $k$ tests.

## Table 1 — Main Results (per-experiment late-vs-early learning)

| Experiment | Model | N | Last-10 mean | 95 % CI | Cohen's *d* | *d* 95 % CI | p (raw) | p (Bonf.) |
|:-----------|:------|--:|-------------:|:--------|------------:|:------------|--------:|----------:|
| campaign_v2_w1_deepseek-v31-base |  | 6 | 0.573 | [0.438, 0.698] | — | — | — | — |
| campaign_v2_w1_gpt-oss-120b |  | 6 | 0.875 | [0.782, 0.959] | — | — | — | — |
| campaign_v2_w1_kimi-k2-thinking |  | 6 | 0.958 | [0.896, 1.000] | — | — | — | — |
| campaign_v2_w1_kimi-k25 |  | 6 | 0.625 | [0.385, 0.865] | — | — | — | — |
| campaign_v2_w1_llama31-70b-base |  | 6 | 0.364 | [0.240, 0.489] | — | — | — | — |
| campaign_v2_w1_llama31-8b-base |  | 6 | 0.115 | [0.052, 0.167] | — | — | — | — |
| campaign_v2_w1_qwen3-8b-base |  | 30 | 0.856 | [0.769, 0.931] | 0.131 | [-0.747, 1.008] | 0.818 | 1.000 |
| campaign_v2_w1_qwen35-397b |  | 6 | 0.979 | [0.938, 1.000] | — | — | — | — |
| campaign_v2_w2_qwen3-8b_G16 |  | 6 | 0.380 | [0.224, 0.557] | — | — | — | — |
| campaign_v2_w2_qwen3-8b_G2 |  | 6 | 0.375 | [0.208, 0.500] | — | — | — | — |
| campaign_v2_w2_qwen3-8b_G32 |  | 3 | 0.438 | [0.422, 0.453] | — | — | — | — |
| campaign_v2_w2_qwen3-8b_G4 |  | 6 | 0.521 | [0.375, 0.646] | — | — | — | — |
| campaign_v2_w3_dsv31_s123 |  | 1 | 0.875 | [0.875, 0.875] | — | — | — | — |
| campaign_v2_w3_dsv31_s456 |  | 1 | 1.000 | [1.000, 1.000] | — | — | — | — |
| cleanrl_ppo_math_s1024 | cleanrl-ppo | 19 | 0.003 | [0.001, 0.005] | — | — | — | — |
| cleanrl_ppo_math_s123 | cleanrl-ppo | 19 | 0.002 | [0.001, 0.004] | — | — | — | — |
| cleanrl_ppo_math_s42 | cleanrl-ppo | 19 | 0.011 | [0.008, 0.014] | — | — | — | — |
| cleanrl_ppo_math_s456 | cleanrl-ppo | 19 | 0.008 | [0.005, 0.011] | — | — | — | — |
| cleanrl_ppo_math_s789 | cleanrl-ppo | 19 | 0.009 | [0.007, 0.011] | — | — | — | — |
| cross_tool_llama-8b-inst | llama-8b-inst | 30 | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 1.000 | 1.000 |
| cross_tool_qwen3-32b | qwen3-32b | 30 | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 1.000 | 1.000 |
| frontier_gsm8k_deepseek-v3.1 | deepseek-v3.1 | 20 | 0.850 | [0.750, 0.938] | 0.087 | [-0.790, 0.964] | 0.694 | 1.000 |
| frontier_gsm8k_nemotron-120b | nemotron-120b | 20 | 0.163 | [0.050, 0.287] | -0.097 | [-0.974, 0.780] | 0.868 | 1.000 |
| frontier_gsm8k_qwen3-235b | qwen3-235b-moe | 4 | 1.000 | [1.000, 1.000] | — | — | — | — |
| modal_trl_trl_llama32_1b |  | 30 | 0.362 | [0.200, 0.537] | 0.817 | [-0.095, 1.730] | 0.084 | 1.000 |
| modal_trl_trl_llama32_3b |  | 30 | 0.713 | [0.550, 0.850] | -0.328 | [-1.210, 0.554] | 0.643 | 1.000 |
| modal_trl_trl_qwen3_8b |  | 30 | 0.050 | [0.000, 0.125] | 0.268 | [-0.612, 1.149] | 0.957 | 1.000 |
| moe_gsm8k_qwen3-30b-inst | qwen3-30b-moe-inst | 3 | 1.000 | [1.000, 1.000] | — | — | — | — |
| moe_gsm8k_qwen3-30b-moe | qwen3-30b-moe | 5 | 0.325 | [0.162, 0.463] | — | — | — | — |
| ppo_llama-8b-inst | llama-8b-inst | 30 | 0.950 | [0.875, 1.000] | -0.268 | [-1.149, 0.612] | 0.583 | 1.000 |
| ppo_qwen3-8b | qwen3-8b | 30 | 0.350 | [0.175, 0.525] | 0.079 | [-0.797, 0.956] | 0.782 | 1.000 |
| sb3_ppo_math_s1024 | sb3-ppo | 49 | 0.010 | [0.006, 0.015] | 1.374 | [0.400, 2.349] | 0.005 | 0.126 |
| sb3_ppo_math_s123 | sb3-ppo | 49 | 0.008 | [0.004, 0.011] | -0.437 | [-1.323, 0.450] | 0.390 | 1.000 |
| sb3_ppo_math_s42 | sb3-ppo | 49 | 0.008 | [0.005, 0.011] | 0.268 | [-0.612, 1.149] | 0.455 | 1.000 |
| sb3_ppo_math_s456 | sb3-ppo | 49 | 0.007 | [0.002, 0.011] | -0.223 | [-1.103, 0.656] | 0.459 | 1.000 |
| sb3_ppo_math_s789 | sb3-ppo | 49 | 0.010 | [0.006, 0.014] | 0.509 | [-0.382, 1.399] | 0.254 | 1.000 |
| scale_gsm8k_llama-8b-inst | llama-8b-inst | 30 | 0.844 | [0.731, 0.944] | -0.526 | [-1.417, 0.366] | 0.271 | 1.000 |
| scale_gsm8k_qwen3-32b | qwen3-32b | 3 | 0.250 | [0.125, 0.312] | — | — | — | — |
| scale_gsm8k_qwen3-8b | qwen3-8b | 30 | 0.344 | [0.263, 0.431] | 0.655 | [-0.245, 1.555] | 0.108 | 1.000 |
| scale_gsm8k_qwen3.5-27b | qwen3.5-27b | 3 | 0.437 | [0.000, 0.750] | — | — | — | — |
| scale_gsm8k_qwen3.5-4b | qwen3.5-4b | 30 | 0.850 | [0.719, 0.969] | 0.178 | [-0.700, 1.056] | 0.633 | 1.000 |
| tianshou_ppo_math_s1024 | tianshou-ppo | 20 | 0.008 | [0.004, 0.011] | 0.000 | [-0.877, 0.877] | 0.938 | 1.000 |
| tianshou_ppo_math_s123 | tianshou-ppo | 20 | 0.005 | [0.002, 0.009] | 0.460 | [-0.428, 1.348] | 0.390 | 1.000 |
| tianshou_ppo_math_s42 | tianshou-ppo | 20 | 0.014 | [0.009, 0.018] | 1.810 | [0.769, 2.850] | 0.002 | 0.048 |
| tianshou_ppo_math_s456 | tianshou-ppo | 20 | 0.010 | [0.007, 0.014] | 0.522 | [-0.369, 1.414] | 0.246 | 1.000 |
| tianshou_ppo_math_s789 | tianshou-ppo | 20 | 0.006 | [0.003, 0.009] | 0.000 | [-0.877, 0.877] | 0.813 | 1.000 |

## Table 2 — Cross-Library Arithmetic (Bonferroni across k = 4, reference = TRL (GRPO))

| Library | N | Mean | 95 % CI | Cohen's *d* vs ref | *d* 95 % CI | p (raw) | p (Bonf.) |
|:--------|--:|-----:|:--------|-------------------:|:------------|--------:|----------:|
| TRL (GRPO) | 5 | 0.734 | [0.673, 0.782] | 0.00 | [0.00, 0.00] | — | — |
| Tinker (GRPO) | 5 | 0.999 | [0.997, 1.000] | -5.32 | [-7.96, -2.68] | 0.001 | 0.004 |
| SB3 (PPO) | 5 | 0.009 | [0.007, 0.010] | 14.59 | [8.08, 21.10] | <0.001 | <0.001 |
| CleanRL (PPO) | 5 | 0.007 | [0.003, 0.010] | 14.61 | [8.09, 21.13] | <0.001 | <0.001 |
| Tianshou (PPO) | 5 | 0.008 | [0.006, 0.011] | 14.58 | [8.07, 21.09] | <0.001 | <0.001 |

## Table 3 — GSM8K Scaling (Bonferroni across k = 7)

| Model | Baseline | Post-RL | Δ | 95 % CI(Δ) | Cohen's *d* | *d* 95 % CI | p (raw) | p (Bonf.) |
|:------|--------:|--------:|---:|:-----------|------------:|:------------|--------:|----------:|
| Qwen3-0.6B | 59.6 | 73.5 | +13.9 | [11.9, 15.9] | 5.18 | [1.85, 8.51] | <0.001 | 0.002 |
| Llama-3.2-1B | 44.4 | 56.8 | +12.4 | [10.2, 15.0] | 3.96 | [1.35, 6.57] | <0.001 | 0.006 |
| Llama-3.2-3B | 77.7 | 85.3 | +7.6 | [6.0, 9.3] | 3.78 | [1.28, 6.28] | 0.001 | 0.008 |
| Qwen3-4B | 87.8 | 93.1 | +5.3 | [4.1, 6.5] | 3.39 | [1.11, 5.66] | 0.002 | 0.011 |
| Qwen3-8B | 89.8 | 94.2 | +4.4 | [3.6, 5.3] | 3.94 | [1.34, 6.53] | <0.001 | 0.006 |
| Qwen3-14B | 92.5 | 95.8 | +3.3 | [2.5, 3.8] | 3.69 | [1.24, 6.14] | 0.001 | 0.008 |
| Qwen3-30B-A3B (MoE) | 91.8 | 95.4 | +3.6 | [2.8, 4.5] | 3.22 | [1.04, 5.40] | 0.002 | 0.014 |

## Table 4 — PPO vs GRPO (Bonferroni across k = 2 model pairs)

| Model | GRPO mean [95 % CI] | PPO mean [95 % CI] | Δ (GRPO−PPO) [95 % CI] | Cohen's *d* | *d* 95 % CI | Welch p (raw) | Welch p (Bonf.) | MW p (Bonf.) |
|:------|:--------------------|:-------------------|:-----------------------|------------:|:------------|--------------:|----------------:|-------------:|
| Qwen3-8B | 0.285 [0.225, 0.348] | 0.283 [0.183, 0.383] | +0.002 [-0.119, 0.119] | +0.01 | [-0.50, 0.51] | 0.973 | 1.000 | 1.000 |
| Llama-3.1-8B-Inst | 0.869 [0.806, 0.923] | 0.950 [0.908, 0.983] | -0.081 [-0.154, -0.010] | -0.56 | [-1.08, -0.04] | 0.035 | 0.070 | 0.012 |

## TRL-GRPO Cross-Seed Baseline (Qwen2.5-0.5B, 5 seeds)

- **Mean accuracy:** 0.734 (95 % CI [0.672, 0.783])
- **SD:** 0.070 · **CV:** 0.096
- **One-sample t vs 0.5:** t = 7.44, p = 0.002, Cohen's *d* = 3.33

## Family-Wide Bonferroni across k = 38 comparisons

| Source | Comparison | Cohen's *d* | p (raw) | p (Bonf., global) | p (BH, global) |
|:-------|:-----------|------------:|--------:|------------------:|---------------:|
| Table 1 | Late-10 vs Early-10 reward (campaign_v2_w1_qwen3-8b-base) | +0.13 | 0.818 | 1.000 | 0.971 |
| Table 1 | Late-10 vs Early-10 reward (cross_tool_llama-8b-inst) | +0.00 | 1.000 | 1.000 | 1.000 |
| Table 1 | Late-10 vs Early-10 reward (cross_tool_qwen3-32b) | +0.00 | 1.000 | 1.000 | 1.000 |
| Table 1 | Late-10 vs Early-10 reward (frontier_gsm8k_deepseek-v3.1) | +0.09 | 0.694 | 1.000 | 0.929 |
| Table 1 | Late-10 vs Early-10 reward (frontier_gsm8k_nemotron-120b) | -0.10 | 0.868 | 1.000 | 1.000 |
| Table 1 | Late-10 vs Early-10 reward (modal_trl_trl_llama32_1b) | +0.82 | 0.084 | 1.000 | 0.199 |
| Table 1 | Late-10 vs Early-10 reward (modal_trl_trl_llama32_3b) | -0.33 | 0.643 | 1.000 | 0.905 |
| Table 1 | Late-10 vs Early-10 reward (modal_trl_trl_qwen3_8b) | +0.27 | 0.957 | 1.000 | 1.000 |
| Table 1 | Late-10 vs Early-10 reward (ppo_llama-8b-inst) | -0.27 | 0.583 | 1.000 | 0.886 |
| Table 1 | Late-10 vs Early-10 reward (ppo_qwen3-8b) | +0.08 | 0.782 | 1.000 | 0.971 |
| Table 1 | Late-10 vs Early-10 reward (sb3_ppo_math_s1024) | +1.37 | 0.005 | 0.208 | 0.016 |
| Table 1 | Late-10 vs Early-10 reward (sb3_ppo_math_s123) | -0.44 | 0.390 | 1.000 | 0.674 |
| Table 1 | Late-10 vs Early-10 reward (sb3_ppo_math_s42) | +0.27 | 0.455 | 1.000 | 0.727 |
| Table 1 | Late-10 vs Early-10 reward (sb3_ppo_math_s456) | -0.22 | 0.459 | 1.000 | 0.727 |
| Table 1 | Late-10 vs Early-10 reward (sb3_ppo_math_s789) | +0.51 | 0.254 | 1.000 | 0.508 |
| Table 1 | Late-10 vs Early-10 reward (scale_gsm8k_llama-8b-inst) | -0.53 | 0.271 | 1.000 | 0.516 |
| Table 1 | Late-10 vs Early-10 reward (scale_gsm8k_qwen3-8b) | +0.66 | 0.108 | 1.000 | 0.241 |
| Table 1 | Late-10 vs Early-10 reward (scale_gsm8k_qwen3.5-4b) | +0.18 | 0.633 | 1.000 | 0.905 |
| Table 1 | Late-10 vs Early-10 reward (tianshou_ppo_math_s1024) | +0.00 | 0.938 | 1.000 | 1.000 |
| Table 1 | Late-10 vs Early-10 reward (tianshou_ppo_math_s123) | +0.46 | 0.390 | 1.000 | 0.674 |
| Table 1 | Late-10 vs Early-10 reward (tianshou_ppo_math_s42) | +1.81 | 0.002 | 0.080 | 0.007 |
| Table 1 | Late-10 vs Early-10 reward (tianshou_ppo_math_s456) | +0.52 | 0.246 | 1.000 | 0.508 |
| Table 1 | Late-10 vs Early-10 reward (tianshou_ppo_math_s789) | +0.00 | 0.813 | 1.000 | 0.971 |
| Table 2 | Tinker (GRPO) vs TRL (GRPO) (final arithmetic accuracy) | -5.32 | 0.001 | 0.041 | 0.005 |
| Table 2 | SB3 (PPO) vs TRL (GRPO) (final arithmetic accuracy) | +14.59 | <0.001 | <0.001 | <0.001 |
| Table 2 | CleanRL (PPO) vs TRL (GRPO) (final arithmetic accuracy) | +14.61 | <0.001 | <0.001 | <0.001 |
| Table 2 | Tianshou (PPO) vs TRL (GRPO) (final arithmetic accuracy) | +14.58 | <0.001 | <0.001 | <0.001 |
| Table 3 | Qwen3-0.6B: post-RL vs baseline (GSM8K) | +5.18 | <0.001 | 0.012 | 0.003 |
| Table 3 | Llama-3.2-1B: post-RL vs baseline (GSM8K) | +3.96 | <0.001 | 0.034 | 0.005 |
| Table 3 | Llama-3.2-3B: post-RL vs baseline (GSM8K) | +3.78 | 0.001 | 0.041 | 0.005 |
| Table 3 | Qwen3-4B: post-RL vs baseline (GSM8K) | +3.39 | 0.002 | 0.062 | 0.006 |
| Table 3 | Qwen3-8B: post-RL vs baseline (GSM8K) | +3.94 | <0.001 | 0.035 | 0.005 |
| Table 3 | Qwen3-14B: post-RL vs baseline (GSM8K) | +3.69 | 0.001 | 0.045 | 0.005 |
| Table 3 | Qwen3-30B-A3B (MoE): post-RL vs baseline (GSM8K) | +3.22 | 0.002 | 0.075 | 0.007 |
| Table 4 | Qwen3-8B: PPO vs GRPO (full trace, n=30) | +0.01 | 0.973 | 1.000 | 1.000 |
| Table 4 | Llama-3.1-8B-Inst: PPO vs GRPO (full trace, n=30) | -0.56 | 0.035 | 1.000 | 0.088 |
| Table 4 | Qwen3-8B: PPO vs GRPO (Mann-Whitney U) | +0.01 | 0.709 | 1.000 | 0.929 |
| Table 4 | Llama-3.1-8B-Inst: PPO vs GRPO (Mann-Whitney U) | -0.56 | 0.006 | 0.219 | 0.016 |

## Protocol Notes

1. **Determinism.** All bootstrap draws use `np.random.SeedSequence(MASTER_SEED=20260506).spawn_key`; each call site derives an independent stream from a string tag.
2. **Bootstrap.** Percentile CIs with B = 10,000; resamples drawn with replacement from the empirical trace.
3. **Effect size.** Cohen's *d* uses pooled (Welch-neutral) SD; Hedges' *g* corrects for small-sample bias. Analytical 95 % CIs follow Hedges–Olkin (1985).
4. **Multiple comparisons.** We report Bonferroni inside each table (local family) and globally across the full comparison set. BH-FDR is reported as a less conservative alternative.
5. **Synthesized seed clouds.** Tables 2-3 recompute variability from the published mean ± SE (with n = 5 seeds) by generating a mean-zero, variance-matched cloud for p-value and *d* estimation; the cloud is deterministic in MASTER_SEED.
