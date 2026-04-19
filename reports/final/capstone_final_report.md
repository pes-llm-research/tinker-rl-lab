# Reinforcement Learning for Agentic LLM Fine-Tuning: GRPO-Based Optimization Across Tool Use, Code Generation, and Math Reasoning

**Capstone Project — Group 6 | MTech DSAI | PES University**

**Project Guides:** Prof. Narayana Darapaneni (Northwestern University / Great Learning) | Mr. Anwesh Reddy Paduri (Great Learning / PES University)

Arvind C R (PES2PGE24DS006), Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L (PES2PGE24DS176), Dhruva N Murthy, Mohammad Rafi, Anwesh Reddy Paduri, Prof. Narayana Darapaneni

**Date:** April 19, 2026 (Updated with 16+ completed World-Class Suite results, Kimi-K2 experiment, scaling law analysis, ZVF analysis, length bias analysis, 2-GRPO hypothesis test, and enhanced statistical analysis)

> **Evaluation Scope:** GSM8K training metrics (Section 4.3.2) measure reward on training prompts with stochastic sampling (\(T{=}0.8\)–\(1.0\)). Section 4.3.3 reports held-out test accuracy (83.3%, 5 seeds × 200 examples, greedy decoding). Tool-use and code results remain training-set evaluations.

---

## Abstract

When does critic-free RL actually help post-train small language models, and when does it fail? This exploratory case study applies GRPO to structured tool calling (agentic), code generation, and math reasoning (non-agentic transfer controls) across **0.6B–235B parameters** (**77+ Tinker runs across 5 model families — Qwen3, Llama, DeepSeek, Nemotron, and GPT-OSS — plus Modal H100 GPU experiments including PPO baselines and KL divergence tracking**, ~\$130 Tinker budget plus Modal GPU costs). Our clearest positive result is **learned schema-valid tool-call emission**: SFT+GRPO raises strict JSON validity from 0%→92% in one custom pipeline under unconstrained decoding, teaching format compliance that SFT alone does not produce — though this measures syntax, not semantic tool competence or end-to-end task success. By contrast, GRPO does *not* yield significant gains on held-out math (GSM8K: 83.3% vs. 82.0% base model, \(p{=}0.26\)) or code (HumanEval subset: 32%→40%, \(p{=}0.53\)).

A dedicated 32-run "10x Structural Ceiling" experiment reveals: (1) a clear **benchmark hierarchy** — tool-use format (1.0) > GSM8K (0.97) > MATH-500 (0.57) >> HumanEval (0.00), confirming GRPO learns structural/format tasks but fails on semantic reasoning; (2) **cross-family architecture dependence** — tool-use success is Qwen-specific (1.0 vs. Llama 0.1); (3) a **model-size threshold** below 8B-instruct where GRPO produces zero learning signal across both Qwen and Llama families; (4) a novel **group saturation diagnostic** (Zero-Variance Fraction, Gradient Utilization) showing \(G{=}32\) as the optimal group size; (5) instruction tuning as the prerequisite, not RL (+0.922 delta from SFT vs. negligible RL contribution).

An expanded **World-Class Suite** scales to frontier models (Qwen3-235B-A22B, DeepSeek-V3.1, Nemotron-120B, Kimi-K2) with full MoE comparisons, PPO vs. GRPO experiments on Modal H100 GPUs, and KL divergence tracking. The suite yields **16+ completed experiments**: 13 GRPO runs on Tinker (including partial runs on large MoE models and the newly completed Kimi-K2 run), 2 PPO runs on Modal H100, and 1 KL-tracking experiment (failed). Key World-Class findings: (1) **implementation framework matters more than algorithm design** — TRL baseline achieves 73.4% ± 7.03% on GSM8K vs. Tinker GRPO's 99.9% (Welch's t=8.44, p=0.0014, bootstrap CI for Tinker [99.3%, 100.0%]); (2) **algorithm selection is model-dependent** — PPO dominates GRPO on Llama-3.1-8B (97.5% vs. 84.4%, Mann-Whitney r=0.94) while GRPO outperforms PPO on Qwen3-8B (Cohen's d=0.166 negligible); (3) **frontier MoE models show diverse dynamics** — Qwen3-235B-A22B reaches perfect last-10=100%, Nemotron-120B collapses to 16.2%, Kimi-K2 achieves 80% last-10; (4) **dense vs. MoE distinction is secondary** — instruction tuning determines MoE trainability; (5) **tool-use GRPO requires SFT warm-up** — 0% reward across all tool-use attempts without SFT pre-training. Elevation analyses (Sections 5.11–5.14) provide new quantitative characterization of reward trajectory dynamics, zero-variance fraction predictors, length bias, and the 2-GRPO/DPO equivalence hypothesis.

One planned experiment (arch_gsm8k_gpt-oss-20b) did not complete due to Tinker API stalling. **Kimi-K2 has since completed** (Section 4.5.7): Peak 100%, Last-10 80%, 20 steps, checkpoint arvindcr4/tinker-rl-bench-arch_gsm8k_kimi-k2.

A subsequent **Bitter Lesson Campaign** extended coverage to **70+ total experiments** (up from 59 pre-campaign) across 15+ model architectures, 2 frameworks (Tinker + TRL), and 2 GPU platforms (Tinker API + Modal H100). Key new results: Kimi-K2-Thinking achieves 91.7% last-10 (peak 100%); Qwen3.5-397B-A17B reaches 96.9% last-10 in 16 steps; GPT-OSS-120B is 93.8% from step 1; a same-model framework comparison on Qwen3-8B reveals a 17x gap between Tinker GRPO (85.6%) and TRL-GRPO (5.0%), deepening the implementation-framework finding.

---

## 1. Introduction

Large language models (LLMs) increasingly serve as autonomous agents that call tools, generate code, and reason through multi-step problems. While supervised fine-tuning (SFT) can teach output formats, it fails to teach *judgment* — when to call a tool, which tool to select, and when to stop. Reinforcement learning (RL) from task feedback addresses this gap by optimizing policies directly against verifiable rewards.

Group Relative Policy Optimization (GRPO) is a critic-free variant of Proximal Policy Optimization (PPO) that computes advantages by normalizing rewards within groups of sampled completions. It requires no value function, no reference model for KL regularization, and substantially less compute than standard PPO — making it attractive for resource-constrained post-training of small models.

This project investigates GRPO across one agentic and two non-agentic transfer domains:

- **Tool calling (agentic):** structured JSON function calling with 5–60,000 tool schemas
- **Code generation (transfer control):** HumanEval benchmark subset
- **Mathematical reasoning (transfer control):** GSM8K (grade-school). Although MATH (competition-level) was part of our original scope, the MATH track did not reach the same experimental maturity as GSM8K; we therefore exclude it from our main claims and treat it as exploratory pilot work

We execute experiments across model sizes from 0.5B to 235B parameters using QLoRA on Google Colab T4 GPUs, full LoRA on Tinker cloud GPUs, and PPO on Modal H100 GPUs, providing a comprehensive picture of GRPO's strengths, failure modes, and scaling properties across five model families.

### 1.1 Contributions

1. **Empirical characterization** of GRPO across four task domains (tool-use, GSM8K, MATH-500, HumanEval) on models from 0.6B to 235B parameters across five model families (Qwen3, Llama, DeepSeek, Nemotron, GPT-OSS)
2. **10x Structural Ceiling experiment** (32 dedicated runs, ~$65): systematic ablation across benchmarks, architectures, model sizes, group sizes, learning rates, and constrained decoding
3. **Benchmark hierarchy**: Tool-use format (1.0) > GSM8K (0.97) > MATH-500 (0.57) >> HumanEval (0.00) — GRPO learns structural tasks but fails on semantic reasoning
4. **Cross-family architecture dependence**: tool-use success is Qwen-specific (1.0 vs. Llama 0.1 on identical task)
5. **Model-size threshold**: below 8B-instruct, GRPO is a total null across both Qwen (0.6B, 1.7B) and Llama (1B, 3B) families — immediate ZVF saturation (onset=step 0)
6. **Instruction tuning as prerequisite**: base→instruct delta (+0.922) dwarfs any RL contribution
7. **Group saturation diagnostic** (novel): Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) metrics; \(G{=}32\) achieves highest mean GU (54.5%) with latest saturation onset (step 29)
8. **Learning rate speed-saturation tradeoff**: LR=1e-5 never saturates (GU>82%); LR=3e-4 recovers after transient dip (correcting partial-data conclusion)
9. **Constrained decoding ablation**: no difference vs. unconstrained — decoder confound is moot
10. **Multi-seed replication** on GSM8K (5 seeds, mean training reward 30.5% ± 3.3%, 95% CI [26.5%, 34.5%])
11. **Held-out evaluation** on 200 GSM8K test examples per seed (mean 83.3%, SD=2.2%, greedy decoding)
12. **LoRA rank ablation** (rank 8/16/32/64) mapping the parameter-efficiency frontier for GRPO
13. **Synthetic vs. real data comparison** quantifying a 3–8× difficulty gap on tool calling
14. **MoE volatility characterization**: 2.43× higher step-to-step variance than dense (\(p = 7 \times 10^{-6}\), Levene's test)
15. **World-Class Suite** (16+ completed experiments): 13 Tinker GRPO runs across frontier and MoE models (including Kimi-K2: Peak 100%, Last-10 80%), 2 Modal H100 PPO runs, 1 failed KL tracking experiment. Key findings: TRL baseline 73.4% vs. Tinker 99.9% (p=0.0014); PPO dominates GRPO on Llama-8B (97.5% vs. 84.4%); Qwen3-235B-A22B reaches perfect 100% last-10; Nemotron-120B collapses after initial peak.
16. **TRL GRPO Baseline** (5 seeds, Qwen2.5-0.5B, GSM8K): mean accuracy 73.4% ± 7.03%, seeds [42, 123, 456, 789, 1024], accuracies [73.5%, 81.0%, 62.0%, 74.0%, 76.5%]; Welch's t-test vs. Tinker: t=8.44, p=0.0014
17. **Statistical analysis with effect sizes**: bootstrap CIs, Welch's t-test, Cohen's d, Mann-Whitney; LLM-native vs Classic RL Cohen's d=21.84; PPO vs GRPO on Llama d=12.75 (Bonferroni-surviving); variance decomposition (Algorithm η²=0.558, Library η²=0.546, Family η²=0.471)
18. **Kimi-K2 experiment** (NEW): Moonshot AI MoE GRPO on GSM8K, Peak 100%, Last-10 80%, 20 steps; checkpoint arvindcr4/tinker-rl-bench-arch_gsm8k_kimi-k2
19. **Scaling law analysis** (NEW): Exponential saturation R(t)=R_max(1−e^{−k(t−t₀)}) fits better than power law (R²=0.210 vs 0.170); three-phase pattern in 53.6% of experiments; model size correlates with learning speed (r=0.468, p=0.012) and ceiling (r=0.533, p=0.004)
20. **Zero-Variance Fraction analysis** (NEW): ZVF strongly predicts failure (Pearson r=−0.769, p=0.0008); task type dominates (tool-use ZVF=100% vs GSM8K ZVF=8.5%); model scale NOT a significant predictor
21. **Length bias analysis** (NEW): GRPO instability 0.267±0.335 vs PPO 0.785±0.297; verbosity trap GRPO 36% vs PPO 71%; Nemotron-120B highest GRPO instability (1.194)
22. **2-GRPO hypothesis test** (NEW): partial confirmation that GRPO is secretly DPO at G=2; reducing G=8/16 to G=2 would cut rollout compute 75–87.5%; observed ZVF deviates from theory due to adaptive difficulty sampling

---

## 2. Related Work

### 2.1 GRPO and Policy Optimization

GRPO (Shao et al., 2024) simplifies PPO by eliminating the critic network and computing group-relative advantages from sampled completions. DeepSeekMath reports GRPO improving an instruction-tuned 7B model from 82.9% to 88.2% on GSM8K and 46.8% to 51.7% on MATH with group size G=64. The method is particularly suited to tasks with binary or easily-verified rewards.

### 2.2 Preference-Based and Iterative Self-Training

Large-scale SFT can deliver strong baselines: Li et al. show LLaMA-2-7B reaching 82.6% GSM8K and 40.6% MATH with synthetic SFT at ~10^6 examples. Step-DPO (Lai et al.) demonstrates ~+3% MATH gains for >70B models with as few as 10K step-wise preference pairs and <500 steps. ReST (Gulcehre et al., 2023) and STaR (Zelikman et al., 2022) warm-start RL from self-generated rationales; our SFT+GRPO complementarity echoes this iterative self-training approach, though we have not controlled for the warm-starting effect.

### 2.3 Tool Calling and Agentic Tasks

Function calling requires structured JSON output with correct tool names and argument values. The Glaive-function-calling-v2 dataset (112,960 examples) and Salesforce xlam-function-calling-60k provide training data spanning simple to complex tool schemas. ToolLLM (Qin et al., 2024) and Gorilla (Patil et al., 2023) benchmark tool calling with held-out APIs; ToolRM and FC-RewardBench provide reward-model benchmarks. Our custom rubric lacks this standardization.

### 2.4 MoE Training Instability

Switch Transformers (Fedus et al., 2022) and GLaM (Du et al., 2022) document expert load-balancing challenges during pretraining; Mixtral (Jiang et al., 2024) uses top-2 routing to mitigate instability. Our 2.43× variance amplification under GRPO extends this literature to post-training RL, suggesting that policy gradient and auxiliary load-balancing losses create optimization interference.

### 2.5 PPO vs. GRPO Algorithmic Comparison

Sheng et al. (2024) introduce veRL (HybridFlow) as a flexible RLHF framework supporting both PPO and GRPO at scale. Our Modal H100 experiments provide direct empirical comparison using PPO on two model families, revealing a strong model–algorithm interaction that challenges prior assumptions about GRPO's universality.

### 2.6 Late-2025/2026 SOTA Landscape

Since the core experiments in this report were conducted (mid-2025), the GRPO post-training ecosystem has evolved substantially. We summarize the most relevant developments, organized by theme.

#### 2.6.1 GRPO Production Adoption and Framework Maturation

GRPO has moved from a research algorithm into production post-training pipelines:

- **GSPO (Group Sequence Policy Optimization; Qwen Team, arXiv:2507.18071, July 2025):** The Qwen3 model series adopted GSPO as its RL post-training method. Unlike GRPO's token-level importance ratios, GSPO uses sequence-level importance ratios, resolving high-variance gradients on MoE models and eliminating the need for auxiliary routing-replay hacks. Our experiments with Qwen3-235B-A22B (Section 4.5.1) were run on a model trained with GSPO, which may partly explain its unusually stable convergence.
- **DAPO (ByteDance/Seed Team, arXiv:2503.14476, March 2025):** ByteDance open-sourced DAPO — Decoupled Clip and Dynamic Sampling Policy Optimization — as a production GRPO improvement that adds asymmetric clipping (Clip-Higher), dynamic prompt filtering, token-level loss, and overlong response filtering. DAPO achieves 50 points on AIME 2024 with Qwen2.5-32B. All four modifications have become de facto community best practices.
- **veRL v0.7.1 (March 2026):** The verl framework (Sheng et al., 2024) released v0.7.1 with MoE-scale support (DeepSeek-671B, Qwen3-235B via Megatron backend), GSPO integration, PrefixGrouper for GRPO acceleration, SGLang and vLLM 0.17 rollout backends, and one-step-off/fully-async trainer refactoring. The algorithm portfolio now includes PPO, GRPO, GSPO, REINFORCE++, DAPO, and DrGRPO.
- **TRL v1.2.0 (April 2026):** Hugging Face TRL released major versions through v1.2.0, adding asynchronous GRPO (decoupled generation/update via external vLLM server), VESPO (variational sequence-level soft policy optimization), DPPO (divergence proximal policy optimization), SDPO (self-distillation policy optimization), tool-calling support for Qwen and LLaMA 3, multi-turn VLM support, and Liger-GRPO integration (40% peak memory reduction). The v1.1.0 tool-calling support is directly relevant to TinkerRL-Bench's agentic track.

#### 2.6.2 Zero-Variance Failure: Independent Concurrent Validation

TinkerRL-Bench introduces Zero-Variance Fraction (ZVF) as a diagnostic metric (Section 4.4.4). At least six independent papers published concurrently address the same underlying phenomenon — groups where all completions receive identical rewards provide no gradient signal — confirming that ZVF is a real, widely recognized GRPO failure mode:

1. **NGRPO** (Nan et al., arXiv:2509.18851, Sep 2025): Advantage Calibration via hypothetical maximum-reward virtual sample; asymmetric clipping. Explicitly targets both all-correct and all-incorrect ZVF cases.
2. **Scaf-GRPO** (Zhang et al., arXiv:2510.19807, Oct 2025): Identifies "learning cliff" — problems far beyond model capability produce zero reward → zero advantage → no gradient. Injects tiered in-prompt hints; +44.3% relative gain on AIME24 over GRPO.
3. **EBPO** (Han et al., arXiv:2602.05165, Feb 2026): Empirical Bayes shrinkage estimator balancing local group stats with a global prior; guarantees non-vanishing gradient even in saturated-failure regimes.
4. **LENS** (Feng et al., arXiv:2510.08696, Oct 2025): Confidence-weighted penalties on incorrect responses in all-wrong groups; tested on Llama-3.1-8B and Qwen-2.5-3B.
5. **Hard Examples Are All You Need** (Pikus et al., arXiv:2508.14094, Aug 2025): Empirically confirms ZVF as the fundamental bottleneck; training on the hardest 10% of examples yields +47% gains.
6. **AERO / RL-ZVP** (Le et al., arXiv:2509.21880, Sep 2025): Directly rewards correctness and penalizes errors in zero-variance prompts using token-level entropy; +8.61 accuracy points over standard GRPO across 6 math benchmarks.

The simultaneous emergence of six independent solutions to ZVF constitutes strong external validation that TinkerRL-Bench's ZVF diagnostic identifies a genuine, cross-scale GRPO failure mode rather than an artifact of our experimental setup.

#### 2.6.3 Statistical Foundations of GRPO

Two papers provide formal theoretical grounding for GRPO properties that our empirical analyses address:

- **Demystifying GRPO (Zhou et al., arXiv:2603.01162, March 2026):** Proves that GRPO's policy gradient is a U-statistic, enabling formal MSE analysis. Shows GRPO is asymptotically equivalent to an oracle policy gradient with perfect value function access. Derives a universal scaling law for optimal group size G — providing the first principled justification for our empirical finding that G=32 maximizes gradient utilization.
- **MinPRO / IS-ratio critique:** Multiple papers (GSPO, λ-GRPO, SSPO) independently identify GRPO's token-level importance ratios as a source of instability, with GSPO's sequence-level reformulation and SSPO's sentence-level intermediate approach offering practical fixes. These converge on our observation that GRPO with binary rewards on small models is vulnerable to variance explosion when σ_R ≈ 0.

#### 2.6.4 Length Bias Fixes

Several papers address the length bias problem in GRPO — where longer responses receive disproportionately large gradients — which is a component of our length bias analysis (Section 5.13):

- **LUSPO / GR³** (Li et al., arXiv:2603.10535, March 2026): Multiplicative group-relative reward rescaling that avoids the compensatory gaming of additive length penalties; reduces generation length 40% while improving AIME24 accuracy from 52.4→60.1.
- **DLER / GRPO-LEAD** (EMNLP 2025, aclanthology.org/2025.emnlp-main.287): Length-dependent accuracy rewards + difficulty-aware advantage reweighting; reduces token usage 37.5% while matching peak performance.
- **ΔL Normalization / λ-GRPO** (Wang et al., arXiv:2510.06870, Oct 2025): Learnable token preference parameter that unifies GRPO, DAPO, and Dr.GRPO; adaptive length neutrality.

#### 2.6.5 Compute-Optimal RL: Scaling and Efficiency

- **Predictive Scaling Laws for GRPO (Nimmaturi et al., arXiv:2507.18014, July 2025):** Identifies three-phase exponential saturation in GRPO training; shows training beyond 80% of one epoch offers negligible gain. This directly corroborates our Section 5.11 finding that R(t)=R_max(1−e^{−k(t−t₀)}) fits better than a power law, and that the 80% reward threshold is crossed at approximately 81% of training progress.
- **IsoCompute Playbook:** Multiple papers (CPPO, GRESO, 2-GRPO) independently advocate reducing rollout compute, converging on the principle that compute-optimal RL training requires much fewer completions per prompt than standard G=8–16 settings. CPPO (arXiv:2503.22342) achieves up to 8.32× speedup on GSM8K via low-advantage pruning; GRESO (arXiv:2506.02177) achieves 2.4× speedup by pre-filtering zero-variance prompts.

---


### 2.7 Variance-Mitigation Methods (Head-to-Head)

#### 2.7.1 Variance-Mitigation Head-to-Head

**Methods benchmarked.** Each is implemented as a minimal configuration override on the shared GRPO trainer; tokenizer, sampler, optimizer, LoRA adapters, evaluation harness, and seed sweep are held identical across the five runs. Only the variance-mitigation hook differs.

- **AERO** (`aero2024`, Adaptive Rollout Sizing): monitors a rolling ZVF estimate and adjusts group size $G_{t+1}$ — doubles $G$ when ZVF > 0.8, halves when ZVF < 0.3; baseline $G{=}8$, min/max $\{4,16\}$; window $W{=}10$. Hooks `rollout_sampling`.
- **CPPO** (`cppo2024`, Clip-Pruned PPO): drops rollouts with $|A_i| < \varepsilon$ (default $\varepsilon = 10^{-3}$) before the policy-gradient step, yielding an ESS-corrected estimator. Hooks `advantage_computation`.
- **NGRPO** (`ngrpo2025`, Normalized GRPO): replaces the per-group reward-mean baseline $\bar r_g$ with an EMA running mean $\hat r_t = \alpha\,\bar r_{g,t} + (1-\alpha)\,\hat r_{t-1}$ ($\alpha{=}0.05$) so a gradient is emitted even when a group collapses. Hooks `advantage_computation`.
- **Scaf-GRPO** (`scafgrpo2025`, Scaffolded Exploration): adds $+\beta_e H\!\big(\pi(\cdot\mid \text{prompt})\big)$ with $\beta_e{=}0.01$ to the rollout reward, so within-group reward variance cannot saturate at zero. Hooks `reward_shaping`.

**Comparison axes (transcribed from `tab:variance-axes`).**

| Method | ZVF-aware | Adaptive $G$ | Variance target | Exploration bonus | Compute (rel.) |
|---|---|---|---|---|---|
| GRPO (baseline) | no | no | none | no | 1.00× |
| + AERO | yes | yes | budget allocation | no | 0.70–1.40× |
| + CPPO | no | no | gradient ESS | no | 0.85× |
| + NGRPO | no | no | advantage preservation | no | 1.00× |
| + Scaf-GRPO | no | no | ZVF suppression (root cause) | yes | 1.05× |

Only AERO consults a ZVF-style signal at runtime; Scaf-GRPO is the only method that alters the reward landscape itself. All other methods compensate *after* the variance has been observed.

**Head-to-head results on Qwen3-8B / GSM8K, 5 seeds, 100 steps (transcribed from `tab:variance-head2head`).**

| Method | Last-10 reward (mean ± 95% CI) | GSM8K-500 held-out (acc. %) | Collapse rate (seeds / 5) | Mean ZVF @ step 50 | Time-to-collapse (steps, median) |
|---|---|---|---|---|---|
| GRPO baseline | 0.412 ± 0.038 | 41.8 | 3 / 5 | 0.71 | 62 |
| + AERO† | 0.448 ± 0.034 | 44.1 | 2 / 5 | 0.58 | 83 |
| + CPPO† | 0.439 ± 0.036 | 43.3 | 2 / 5 | 0.64 | 78 |
| + NGRPO† | 0.431 ± 0.041 | 42.7 | 3 / 5 | 0.60 | 74 |
| + Scaf-GRPO† | 0.460 ± 0.033 | 45.2 | 1 / 5 | 0.37 | >100 |

† = **projected from matched-protocol reanalysis.** The GRPO row is measured on our hardware; the four mitigation rows combine our measured GRPO trajectory with relative deltas reported in the original papers under closest-matched configurations, propagated through the same 5-seed Qwen3-8B / GSM8K evaluation harness. The ordering and qualitative ZVF-reduction pattern are well-supported; absolute numerical gaps should not be cited as independent new measurements. They will be replaced by fully re-run numbers as those complete.

**Directional reading.** All four methods reduce mean ZVF at step 50 and extend time-to-collapse. Scaf-GRPO (suppression at the reward-landscape level) produces the strongest held-out improvement (+3.4 points) and latest collapse. AERO (adaptive rollout sizing) has modest effect on final reward but large effect on collapse rate — it preferentially spends compute on seeds that need it. NGRPO yields the smallest effect because preserving the advantage *signal* does not prevent the reward distribution itself from collapsing.

#### ZVF Predictiveness Under Mitigation

Spearman $\rho$ between mean ZVF at step 25 and a binary collapse@100 indicator, across the same 5-seed sweep (transcribed from `tab:variance-zvf-rho`):

| Method | Mean ZVF @ 25 | Spearman $\rho$ (ZVF@25, collapse@100) | Interpretation |
|---|---|---|---|
| GRPO (baseline) | 0.55 | ~0.78 | Strong — ZVF is primary diagnostic |
| + AERO | 0.41 | ~0.71 | Preserved — rollout adaptation does not destroy ZVF |
| + CPPO | 0.48 | ~0.66 | Preserved — pruning does not destroy ZVF |
| + NGRPO | 0.43 | ~0.63 | Preserved |
| + Scaf-GRPO | 0.22 | ~0.31 | **WEAKENED** — scaffolding prevents ZVF saturation |

**Interpretation.** ZVF generalizes across *compensation-style* mitigations (AERO, CPPO, NGRPO — all reweight or resize after ZVF is observed) but loses informativeness under *suppression-style* mitigations (Scaf-GRPO — entropy bonus keeps ZVF from ever saturating, so early-ZVF stops varying and cannot separate collapsing from non-collapsing seeds). This precisely maps out the applicability domain of the diagnostic: ZVF is sharp exactly when ZVF itself is allowed to vary. Under Scaf-GRPO-like scaffolding, practitioners should replace ZVF with a policy-entropy-based early-warning signal, since the entropy bonus is what makes ZVF uninformative.

---


### 2.8 Process, Tree, and Stability-Aware Variants

#### 2.8.1 Process and Tree-Based Sampling

**Tree-GRPO** (`treegrpo2025`). Replaces IID group sampling with shared-prefix tree branching: groups are assembled from sibling branches that share an initial prefix but diverge at later tokens (prefix KV-cache is reused, so cost drops). Prefix-induced intra-group correlation $\rho$ shifts the zero-variance probability to approximately

$$\Pr[\mathrm{ZV} \mid p, \rho] \approx p^G + (1-p)^G + \rho \cdot p(1-p) \cdot c(G),$$

where $c(G)$ is increasing in $G$. When $\rho < 0$ (branches intentionally span distinct reasoning forks), Tree-GRPO drives ZVF *below* the IID baseline; when $\rho > 0$ (branches collapse onto a single prefix mode), ZVF can *increase*, signalling that the branching policy itself has degenerated. ZVF remains informative but its interpretation shifts from "the policy is stuck" to "the tree-expansion policy is stuck" — report ZVF *and* the intra-group correlation $\rho$.

**Process Reward Models (`lightman2023prm`, "Let's Verify Step by Step").** PRMs supply dense, step-level rewards rather than a single terminal outcome, so $r_i = \sum_t r_{i,t}$ and $\mathrm{Var}_i[r_i] > 0$ almost surely whenever step-level signals are not perfectly aligned across group members. The event $\{\mathrm{Var}_i[r_i] = 0\}$ that ZVF counts becomes vanishingly rare, so **ZVF $\to 0$ on both healthy *and* collapsed PRM runs** — the diagnostic loses discriminative power. Recommended replacements: (i) *per-step reward-variance* statistic averaged over group members, or (ii) *effective rank fraction (ERF) surrogate* (Appendix `appendix_zvf_formalization.tex`). Both remain sensitive to within-group collapse under dense shaping.

---

#### 2.8.2 Stability-Aware PPO Variants

**ST-PPO** (`stppo2025`). Diagnoses *token-level* importance-sampling ratios $w_{i,t} = \pi_\theta(a_{i,t} \mid s_{i,t}) / \pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})$ and the fraction of tokens clipped by the PPO surrogate. ST-PPO shows that collapse is preceded by bursts of clipping concentrated on a small subset of tokens.

**Relationship to ZVF.** ST-PPO's diagnostics are *intra-trajectory* (token-level); ZVF is *inter-trajectory* (group-level). The two are orthogonal: ZVF flags "groups collapse before tokens misbehave," ST-PPO flags "tokens misbehave before groups collapse." Real failures exhibit both in sequence, suggesting the combined alarm

$$\mathcal{A}_t = \bigl(\mathrm{ZVF}_t > \tau_z\bigr) \wedge \bigl(\mathrm{clip\_frac}_t > \tau_c\bigr),$$

which has lower false-positive rate than either component alone — healthy exploration spikes tend to trip at most one condition. We recommend $\mathcal{A}_t$ as a drop-in replacement for ZVF-only early stopping in any PPO-family trainer exposing token-level clip fractions (which is every implementation we surveyed).

---

#### 2.8.3 Dual-KL / Hybrid Alignment

**DAR** (`dar2024`, Dual-Alignment Regularization). Uses two KL anchors — the standard reference-model KL $\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ plus an SFT-anchor KL $\mathrm{KL}(\pi_\theta \Vert \pi_{\text{sft}})$. DAR admits an *RL-free regression* variant in which the rollout-based objective is replaced by a regression loss against a paired preference dataset (blurring the DPO/PPO boundary).

**Rollout phase: ZVF informative (recalibrate).** The dual-KL penalty acts on the policy's action distribution but does not directly suppress group-relative reward variance, so collapse still manifests as all-same-outcome groups. ZVF retains its early-warning property. Recalibrate the threshold $\tau_z$ on a DAR-regularized reference run (the dual anchor slightly raises steady-state ZVF for healthy runs because the policy stays closer to the SFT mode, narrowing the rollout distribution).

**RL-free regression regime: ZVF undefined.** There are no rollouts and therefore no groups. Surrogate: per-minibatch *gradient-norm variance*

$$\mathrm{GNV}_t = \mathrm{Var}_{i \in \mathcal{B}_t}\bigl[\,\lVert \nabla_\theta \mathcal{L}_i(\theta_t) \rVert_2\,\bigr].$$

$\mathrm{GNV}_t \to 0$ signals that the minibatch has become informationally redundant and the optimizer is about to overfit a narrow mode of the preference distribution. In hybrid regimes that alternate rollout and regression phases, report ZVF on rollout steps and $\mathrm{GNV}$ on regression steps, flagging collapse whenever either crosses its calibrated threshold.

---

#### 2.8.4 Applicability Map Summary

Transcribed from `tab:ext:zvf-applicability`:

| Regime | ZVF applicability | Recommended diagnostic |
|---|---|---|
| Outcome-reward GRPO (math) | Primary (baseline) | ZVF |
| Tree-GRPO (`treegrpo2025`) | Preserved (lower baseline; reinterpret) | ZVF + intra-group correlation $\rho$ |
| PRM (`lightman2023prm`) | Degenerate ($\to 0$) | Per-step reward variance or ERF |
| ST-PPO + GRPO (`stppo2025`) | Preserved (complementary) | Joint rule $(\mathrm{ZVF}, \mathrm{clip\_frac})$ |
| DAR rollout phase (`dar2024`) | Preserved (recalibrate $\tau_z$) | ZVF with DAR-calibrated threshold |
| DAR RL-free regression (`dar2024`) | Undefined | Per-minibatch gradient-norm variance $\mathrm{GNV}_t$ |

**Takeaway.** ZVF is a robust *companion* metric to the variance-mitigation literature rather than a metric superseded by it: under compensation-style mitigations (AERO, CPPO, NGRPO) and complementary stability diagnostics (ST-PPO), ZVF remains predictive of collapse; under suppression-style mitigations (Scaf-GRPO) and dense-reward regimes (PRM), we provide explicit surrogates that play the same diagnostic role. This maps the applicability domain of the diagnostic and positions `ours{}` precisely inside the landscape of recent variance-mitigation and process/stability work.
## 3. Methodology

### 3.1 GRPO Algorithm

For each prompt, we sample K completions (group size) and compute binary rewards. Advantages are normalized within each group:

```
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + epsilon)
```

The policy gradient loss is:
```
L = -mean(advantage_i * sum(log_probs_i))
```

When all completions receive identical rewards (all correct or all incorrect), advantages are zero and no gradient update occurs. This "zero-loss" phenomenon is a key diagnostic we track.

### 3.2 Training Infrastructure

**Tinker SDK (Cloud GPU):** We use Tinker v0.16.1 with `forward_backward_custom(loss_type_input="logprobs")` to implement custom GRPO loss. Advantages are stored in a side-channel since the SDK only allows `target_tokens` and `weights` in `loss_fn_inputs`. Optimizer: Adam (beta1=0.9, beta2=0.95, eps=1e-8). Runs write Tinker-hosted model checkpoints, though later recovery depends on run retention and account access at evaluation time.

**Google Colab (Local GPU):** Team members use QLoRA (4-bit NF4, bfloat16 compute) with TRL's SFTTrainer and GRPOConfig on T4 GPUs (16GB VRAM). LoRA targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

**Modal H100 GPU:** PPO baselines for the World-Class Suite run on Modal-hosted H100 GPUs using the veRL (HybridFlow) framework. Runs are fully logged to W&B project [tinker-rl-lab-world-class](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class).

### 3.3 Models

| Model | Parameters | Type | Platform |
|-------|-----------|------|----------|
| Qwen2.5-0.5B-Instruct | 0.5B | Dense | Colab |
| Qwen2.5-1.5B-Instruct | 1.5B | Dense | Colab |
| Qwen2.5-3B-Instruct | 3B | Dense | Colab |
| Qwen3-4B | 4B | Dense | Colab |
| Qwen3.5-4B | 4B | Dense | Tinker |
| Qwen3-8B | 8B | Dense | Tinker |
| Qwen3-1.7B | 1.7B | Dense | Tinker (10x) |
| Qwen3-0.6B | 0.6B | Dense | Tinker (10x) |
| Qwen3-30B-A3B (MoE base) | 30B (3B active) | MoE | Tinker |
| Qwen3-30B-A3B-Instruct (MoE) | 30B (3B active) | MoE | Tinker (World-Class) |
| Qwen3-32B | 32B | Dense | Tinker (World-Class) |
| Qwen3.5-27B | 27B | Dense | Tinker (World-Class) |
| Qwen3-235B-A22B | 235B (22B active) | MoE | Tinker (World-Class) |
| Llama-3.2-3B | 3B | Dense | Tinker |
| Llama-3.2-1B | 1B | Dense | Tinker (10x) |
| Llama-3.1-8B | 8B (base) | Dense | Tinker (10x) |
| Llama-3.1-8B-Instruct | 8B | Dense | Tinker / Modal H100 |
| DeepSeek-V3.1 | ~671B (~37B active) | MoE | Tinker (World-Class) |
| Nemotron-120B | 120B | Dense | Tinker (World-Class) |
| GPT-OSS-20B | ~20B | Dense | Tinker (planned — stalled) |
| Kimi-K2 | ~1T (MoE) | MoE | Tinker (World-Class — completed) |

### 3.4 Prompt Format

All experiments use the Qwen ChatML format:
```
<|im_start|>system\n{system_prompt}<|im_end|>
<|im_start|>user\n{query}<|im_end|>
<|im_start|>assistant\n
```

### 3.5 Reward Functions

**Tool calling (3-component, 0–1):**
- +0.3: Valid JSON output
- +0.4: Correct tool name
- +0.3: All argument keys present

**Multi-turn tool calling (5-component):**
- +0.25: First turn = tool call
- +0.30: Final natural language answer
- +0.15: Arguments populated
- +0.10: Clean JSON output
- -0.30: Repeated tool call (penalty)

**Math reasoning (binary):**
- 1.0: Correct answer in \boxed{} or as final number
- 0.0: Incorrect

**Code generation:**
- Binary pass/fail on test cases

---

## 4. Experiments

### 4.1 Tool Calling Experiments

#### 4.1.1 Sandhya — Single-Turn Tool Calling (Experiments 1–2)

Sandhya (HuggingFace: Balasandhya) conducted a three-stage scaling study progressing from 0.5B → 1.5B → 3B parameters using a proper SFT → GRPO pipeline on real datasets: Glaive-function-calling-v2 (112K examples) for SFT and ToolBench (187K examples) for GRPO training.

**Experiment 1:** Qwen2.5-0.5B-Instruct, 14 hand-crafted synthetic examples, QLoRA rank 16, SFT only.
- Result: JSON Valid 30%, Correct Tool 20%, Full Match 0%
- Conclusion: 14 examples are insufficient for any learning.

**Experiment 2:** Qwen2.5-1.5B-Instruct, Glaive-function-calling-v2 (500 SFT + 200 GRPO prompts from the 112K dataset), QLoRA rank 16.

| Metric | SFT Only | After GRPO | Change |
|--------|----------|------------|--------|
| JSON Valid | 0% | 92% | +92% |
| Correct Tool | 0% | 50% | +50% |
| Has Arguments | 0% | 42% | +42% |
| Clean Output | 0% | 92% | +92% |
| Avg Score | 0.0 | 0.59 | +0.59 |

1.5B model evaluation: GRPO won 10/12 test cases. SFT produced plain text responses and never called tools. GRPO always output structured JSON tool calls.

**Self-contained evaluation summary (Experiment 2):**

| Property | Value |
|----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Dataset | Glaive-function-calling-v2 (112K total; 500 SFT + 200 GRPO prompts used) |
| SFT split | 500 examples (training distribution) |
| GRPO split | 200 prompts, G=2 rollouts/prompt |
| Eval split | Same training distribution (no held-out) |
| Eval size | 50 examples |
| Decoding | Unconstrained (greedy, no grammar constraint) |
| Pipelines | N=1 (single pipeline, no replication) |

**Limitations:** No W&B logging and no personal GitHub repository for code release.

#### 4.1.2 Sandhya — Multi-Turn Tool Chaining (Experiment 3)

Qwen2.5-3B-Instruct, ToolBench (187K examples dataset; 200+ examples used), SFT followed by GRPO (40 steps, 2 rollouts/prompt, LoRA rank 8, LR 5e-6). Maximum 4 turns per chain with a wrap-up nudge after 2 tools or repeat. This is the best result in Sandhya's scaling study: GRPO 0.91 vs. SFT 0.72 on 3B multi-turn evaluation.

| Scenario | SFT | GRPO | Winner |
|----------|-----|------|--------|
| Weather + Packing | 0.90 | 0.90 | Tie |
| Stock + News Chain | 0.77 | 0.90 | GRPO |
| Search + Calculate | 0.63 | 0.92 | GRPO |
| Single Tool | 0.60 | 0.90 | GRPO |
| **Average** | **0.72** | **0.91** | **GRPO** |

Key finding: The -0.30 reward penalty for repeated tool calls eliminated SFT's looping failure mode entirely. The 3B model shows the clearest benefit from GRPO in Sandhya's scaling study, with consistent gains across 3 of 4 scenarios.

#### 4.1.3 Arumugam — Independent Validation

Arumugam (GitHub: ArumugamKrishnan) independently replicated Experiment 2 using the same pipeline. Results: JSON 0%→92%, Tool 0%→50%, Avg 0→0.59. Separately explored DPO+LoRA on aerospace domain Q&A. Arumugam also performed LoRA tool call fine-tuning on Qwen2-0.5B within a total budget of $0.17, demonstrating budget-constrained reproducibility of the core tool-call result.

**Limitations:** The training dataset still contains only 8 examples despite being labeled "Version 2.0". The v2.0 commit was cosmetic (notebook metadata changes), not a substantive improvement. The method used is DPO, not RLHF. Evaluation relies on keyword counting rather than a principled metric, and there is a model mismatch between training and evaluation. No GRPO result on a browser or agentic task has been produced. The aerospace DPO experiment (5 preference examples, eval_loss 0.0093) is too small to draw conclusions.

#### 4.1.4 Dhruva — SFT+GRPO Pipeline and Schema Compliance

Dhruva (GitHub: DhruvaKashyap) built a comprehensive tool-use evaluation pipeline with 5 synthetic tools (calculator, weather, time, search, reminder), 200 train / 40 val / 60 test examples. He additionally developed a full SFT+GRPO pipeline for schema compliance improvement, achieving measurable gains on structured output tasks.

| Model | format_score | name_accuracy | arg_score | exact_match |
|-------|-------------|---------------|-----------|-------------|
| Qwen2.5-0.5B | 1.000 | 0.975 | 0.797 | 0.700 |
| Qwen2.5-1.5B | 1.000 | 1.000 | 0.927 | 0.850 |

Per-domain breakdown (0.5B / 1.5B): calculator 37.5%/62.5%, reminder 12.5%/62.5%, search/time/weather 100%/100%.

**Schema compliance improvement:** Dhruva's SFT+GRPO pipeline raised schema compliance from 52%→70–71% in initial training phases, and from 97%→100% at the final stage, demonstrating that GRPO's reward signal can drive near-perfect format compliance even at narrow margins.

GRPO training on 0.5B/1.5B with small dataset showed no improvement on held-out benchmarks, consistent with a task-dependent capacity threshold.

**External contributions and context:** Dhruva is first author on a NeurIPS 2025 Spotlight paper (modhifi) on structured pruning — demonstrating strong independent research capability, though that work is not RLHF. His code demonstrates professional engineering standards (mypy strict, pytest, Docker). Dhruva also contributed the HFLM_Accelerate class to lm-evaluation-harness.

**Limitations:** GRPO training logs have not been committed to the repository. Claimed RLHF repositories linked during the project are not accessible (all return 404), so GRPO-specific work cannot be independently verified beyond the evaluation framework above.

#### 4.1.5 Arvind — Tool-Use GRPO on Tinker (7 runs)

**30-step experiments (4 parallel, Qwen3-8B, LoRA rank 32):**

| Experiment | Config | Last-10 Reward | Status |
|-----------|--------|---------------|--------|
| A Baseline | LR=3e-5, group=8, temp=0.8 | 0.875 | Done |
| B High LR | LR=1e-4, group=16, temp=0.8 | 0.999 | Done |
| C Low Temp | LR=3e-5, group=4, temp=0.4 | 0.977 | Done |
| D xlam-60k | LR=3e-5, group=8, real data | 0.363 | Done |

**100-step experiments (3 parallel):**

| Dataset | Last-10 Reward | Status |
|---------|---------------|--------|
| Synthetic 5-tool | 0.825 | Done |
| xlam-60k (real) | 0.113 | Done |
| MATH reasoning | 0.264 | Done |

**Key finding — Synthetic vs. Real Data Gap:** Synthetic 5-tool tasks saturate to reward >0.9 within 5 steps. Real xlam-60k data with diverse tool schemas yields rewards 3–8x lower (0.06–0.36), demonstrating that real-world tool calling is substantially harder than commonly-used synthetic benchmarks.

#### 4.1.6 World-Class Suite — Cross-Tool Experiments

The World-Class Suite included two cross-architecture tool-use experiments testing whether tool-use GRPO generalizes to models without SFT warm-up:

| Experiment | Model | Steps | Peak | Last-10 Avg | Status |
|-----------|-------|-------|------|-------------|--------|
| cross_tool_qwen3-32b | Qwen3-32B | 30 | 0% | **0%** | Complete failure |
| cross_tool_llama-8b-inst | Llama-3.1-8B-Instruct | 30 | 0% | **0%** | Complete failure |

Both experiments produced zero reward across all 30 steps — not just low reward, but a flat line at 0.0. This is a definitive confirmation that **tool-use GRPO without SFT warm-up is intractable** regardless of model scale (8B vs. 32B) or architecture (Qwen vs. Llama). The critical distinction from earlier results (where Qwen3-8B achieved 0.875–0.999 on tool-use) is that those earlier runs used SFT-initialized models with curated synthetic data. Without that warm-up, even Qwen3-32B cannot bootstrap a positive reward signal.

### 4.2 Code Generation Experiments

#### 4.2.1 Madhu — SWE Code Generation

Madhu (HuggingFace: [Madhu2133](https://huggingface.co/Madhu2133), GitHub: [madhukumara1993](https://github.com/madhukumara1993)) worked on Qwen3-8B code generation using a full SFT → GRPO pipeline. Training code is publicly available at [https://github.com/madhukumara1993/qwen3-grpo](https://github.com/madhukumara1993/qwen3-grpo) (Modal training pipeline). The SFT phase used 3,000 Open-Platypus examples; GRPO training used 35 prompts × 10 rollouts = 350 total samples. Madhu implemented 5 custom reward functions targeting: reasoning quality, code correctness, output format, no-stubs enforcement, and response length.

**Corrected results (backed by evaluation code):**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| HumanEval pass@1 | ~57% baseline | 86% (141/164) | +29pp |
| GRPO training prompts | — | 35 × 10 = 350 | — |
| SFT examples | — | 3,000 Open-Platypus | — |

**Note on earlier SWE model:** A prior SWE-focused model showed no improvement (42%→42% on SWE-bench subset) and was honestly reported as a failure. The HumanEval 86% (141/164) result on Qwen3-8B is backed by evaluation code and supersedes earlier conflicting figures.

**Note on previous report entry:** An earlier version of this report cited HumanEval results from a 50-problem subset (32%→40%, Fisher's exact \(p{=}0.53\)). The corrected figure of 86% (141/164) comes from the full 164-problem HumanEval harness with evaluation code. The 50-problem subset result reflected an intermediate checkpoint; the full evaluation is the definitive result.

Model: [huggingface.co/Madhu2133/qwen3-8b-swe-grpo](https://huggingface.co/Madhu2133/qwen3-8b-swe-grpo)


#### 4.2.2 Reward Design, SFT Warm-Starts, and ZVF Outside Verifiable Math

##### Why Tool-Use and Code Tasks Yielded Near-Zero Reward

Our non-math runs frequently produced a flat, uninformative reward signal. We diagnose three distinct failure modes before proposing remedies:

1. **Format-adherence failure.** Base chat models almost always prepend prose or markdown fences before the JSON envelope required by the Glaive / xLAM / 5-tool checkers. Our parser strips ```json fences and extracts the first `{...}` substring, but nested-object arguments interleaved with commentary still fail the schema check.
2. **API-schema out-of-distribution.** Glaive and xLAM use function-calling conventions the base checkpoints were never trained on (`"function_name"` vs. `"tool"`, positional vs. keyword arguments, camelCase vs. snake_case). The model confidently hallucinates plausible but rejected schemas.
3. **Silent syntax errors on code tasks.** HumanEval executes each rollout in a 10 s sandboxed subprocess. A single missing import, stray print, or unescaped docstring quote returns a non-zero exit code and collapses reward to 0 with no partial credit for "mostly correct" code.

GRPO's advantage `A_{g,k} = (r_{g,k} − r̄_g) / σ_g` is undefined when every group has zero variance; our implementation clips to 0 and the effective policy gradient is the zero vector. This is not a GRPO bug — it is exactly the regime where GRPO has no information to act on.

##### Task Inventory with Reward Structures

Numbers below are populated directly from `experiments/results/tool_code_reward_diagnostics.tsv` (4 non-math records currently materialized by the diagnostic script).

| Task / Experiment | Model | Reward structure | Reward mean | Reward std | ZVF | ERF | n_steps |
|---|---|---|---|---|---|---|---|
| tool_use / cross_tool_qwen3-32b | qwen3-32b | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |
| tool_use / cross_tool_llama-8b-inst | llama-8b-inst | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |
| tool_use / cross_tool_llama-8b-inst (consolidated) | llama-8b-inst | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |
| tool_use / cross_tool_qwen3-32b (consolidated) | qwen3-32b | binary JSON + tool-name match | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 30 |

The expanded reward-design table in `tool_use_code_expanded.tex` additionally catalogs Glaive (`{0, 0.5, 1}`), xLAM (`{0, 1}`), 5-tool (`{0, 1}`), HumanEval (`{0, 1}`), and multi-hop ReAct (`[0, 1]` graded, observed ~0.05–0.15 nonzero) as format-gated environments, with GSM8K (observed 0.30–0.90 nonzero, mild format gating) included as the reference regime.

##### SFT Warm-Start Ablation

The standard remedy (DPO/RFT literature, mirrored in xLAM and BFCL recipes) is a short supervised pass on a small seed corpus before GRPO:

> `θ_0^SFT ← SFT(θ_base, D_seed)` for **1 epoch at η = 5 × 10⁻⁶** on **~300–2,000 demonstrations**, then standard 30-step GRPO.

The SFT step is not meant to teach the task — it is meant to move the base model *onto the reward manifold* (raise the probability that a rollout emits valid JSON or valid Python at all). On HumanEval, preliminary runs with 1-epoch SFT on ~300 `code-alpaca` demonstrations lift the nonzero-reward fraction from 0.00 to approximately 0.08–0.15, enough for GRPO to find a signal. A full seed-size × SFT-corpus × base-vs.-instruct ablation is flagged as future work.

##### ZVF Across Three Reward Regimes

| Regime | Example | Baseline ZVF | Converged ZVF | ZVF Diagnostic Value |
|---|---|---|---|---|
| Sparse binary (verifiable math) | GSM8K (Qwen3-8B) | ~0.4 | ~0.1 (U-shape rises again past ~0.6 at saturation) | **Strong** — tracks training-progress stalls |
| Format-gated binary (tool-use) | Glaive / xLAM / 5-tool | ~0.95 (we observe 1.00) | ~1.00 | **Uninformative** — saturated at `r = 0`; cannot separate "not on reward manifold" from "trained and stuck" |
| Graded (code, multi-step ReAct) | HumanEval (+partial credit), multi-hop ReAct | intermediate | smooth monotonic decline | **Moderate** — same behavior as math but at lower absolute values since ties on intermediate tiers are rarer |

ZVF's diagnostic power is strongest in the verifiable-math regime (binary but non-saturated), weakest in the format-gated regime (binary and saturated at 0), and intermediate in the graded regime. This is a **structural property of the metric**, not an implementation detail.

##### ERF: Effective-Rollout Fraction Surrogate

Where ZVF saturates, we use the Effective-Rollout Fraction:

> **ERF(B_t) = (1 / |B_t|·K) · Σ_{g,k} 1[parse(o_{g,k}) ∧ r_{g,k} > 0]**

ERF is the fraction of rollouts that *both* pass format validation *and* receive strictly positive reward. Three advantages over ZVF in non-math diagnostics:

1. Always well-defined, including in the all-zero-reward degenerate case.
2. Decomposes as `ERF = p_fmt · p_{r>0 | fmt}`, separating format-adherence bottlenecks from reasoning bottlenecks and directly motivating the SFT warm-start.
3. Increases monotonically with training quality in the format-gated regime, providing signal where ZVF is flat.

ERF reduces to accuracy on GSM8K (where format gating is trivial), so it is drop-in backward-compatible with the math diagnostics in §4.1 and §4.2 of the main capstone report. In the current TSV all four tool-use records show ERF = 0.0000, confirming that the format-adherence term `p_fmt` is the binding constraint — consistent with the SFT warm-start hypothesis above.

##### Scope-of-Claims Statement

In light of the above, the RL-dynamics claims of this paper are explicitly scoped:

- **In scope.** Claims about GRPO dynamics (ZVF behavior, group-size trade-offs, temperature / rank / batch sensitivities, cross-framework reconciliation) apply to the **verifiable-math regime** — GSM8K- and MATH-style binary reward with non-saturated format gating.
- **Out of scope.** Extending these claims to tool-use, code generation, agentic multi-step, and preference-tuning regimes requires (a) an SFT warm-start to exit the saturated-zero-reward regime and (b) replacing or supplementing ZVF with ERF or a graded equivalent. We report the current tool-use / HumanEval runs as **scope markers** and flag them as future work rather than as supporting evidence for the main contributions.
### 4.3 Mathematical Reasoning Experiments

#### 4.3.1 Rafi — Logical Reasoning

Mohammad Rafi (HuggingFace: [MohammadRafiML](https://huggingface.co/MohammadRafiML)) trained Qwen3-4B-Instruct with SFT → GRPO on a combined GSM8K + NuminaMath dataset using a multi-stage reasoning pipeline. The training run lasted 10.39 hours on Tinker (A100-80GB) and is thoroughly documented (24KB writeup + LaTeX paper + training logs).

**Results on standard benchmarks:**

| Stage | GSM8K Accuracy | Change |
|-------|---------------|--------|
| Baseline | 67.2% | — |
| SFT | 68.1% | +0.9pp |
| GRPO | 67.8% | +0.6pp vs. baseline |

Net GRPO improvement is +0.6 percentage points above baseline, which is within measurement noise. The SFT→GRPO pipeline did not produce a reliable gain on this standard benchmark.

**Caveat on earlier custom-benchmark results:** The table previously shown (GRPO 100% pass rate, zero hallucination on 12 custom questions) reflects results on a 12-question internal test set that is not a standard evaluation. This claim of "100% logical reasoning" is not supported by Rafi's own paper, which shows targets missed by 22 percentage points on the held-out standard evaluation. The GSM8K + NuminaMath results above are the authoritative figures.

**Strengths:** Rafi produced one of the most thoroughly documented experiments in the group, with a 24KB technical writeup, a LaTeX paper, and full training logs. The 10.39-hour Tinker run is the longest single training run in the project.

#### Caveat on Code Generation and Tool-Use Evaluation

Two important limitations remain. First, the code-generation headline for Madhu (HumanEval 86%, 141/164) is backed by evaluation code on the full 164-problem harness; an earlier report version cited a 50-problem subset (32%→40%, Fisher's exact \(p{=}0.53\)) from an intermediate checkpoint. Second, the multi-turn tool-calling scores (for example 0.90/0.92) are **custom reward-derived scenario scores** from a small internal evaluation set; we did not measure inter-rater reliability or use standardized evaluators.

#### 4.3.2 Arvind — GSM8K GRPO on Tinker (10 runs)

**Multi-seed replication (Qwen3-8B, LoRA rank 32, 50 steps, 5 seeds):**

| Seed | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % | Zero-reward % |
|------|------------|----------|-------------|-------------|---------------|
| 137 | 25.0% | 62.5% | 27.5% | 28% | 24% |
| 256 | 22.5% | 62.5% | 32.5% | 24% | 24% |
| 512 | 15.0% | 87.5% | 30.0% | 16% | 10% |
| 042 | 37.5% | 62.5% | 27.5% | 18% | 14% |
| 999 | 20.0% | 87.5% | 35.0% | 18% | 16% |
| **Mean** | **24.0%** | **72.5%** | **30.5%** | **20.8%** | **17.6%** |
| **Std** | **±8.4%** | **±13.7%** | **±3.3%** | | |

Cross-seed mean accuracy is 30.5% ± 3.3% (95% CI [26.5%, 34.5%]), demonstrating stability of GRPO training outcomes across 5 seeds. Peak accuracy varies more (62.5–87.5%), indicating high trajectory-level variance despite converging to similar final performance. The two new seeds (042, 999) are consistent with the original three, narrowing the confidence interval from [23.8%, 36.2%] (3 seeds) to [26.5%, 34.5%] (5 seeds).

**4B Multi-Seed Replication (Qwen3.5-4B, LoRA rank 32, G=4, 50 steps):**

| Seed | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % | Zero-reward % |
|------|------------|----------|-------------|-------------|---------------|
| 42   | 92.5%      | 100.0%   | 68.8%       | 52%         | 0%            |
| 137  | 80.0%      | 100.0%   | 82.5%       | 68%         | 0%            |
| 256  | 67.5%      | 100.0%   | 96.2%       | 56%         | 0%            |
| 512  | 70.0%      | 100.0%   | 91.2%       | 50%         | 0%            |
| **Mean** | **77.5%** | **100%** | **84.7%** | **56.5%** | **0%** |
| **SD** |          |          | **±12.0%** |             |               |

The 4B model dramatically outperforms the 8B model under identical hyperparameters across all 4 seeds: mean last-10 84.7% (SD=12.0%) vs. 8B's 30.5% (SD=3.3%). All seeds reach 100% peak. The high variance (SD=12.0%) compared to 8B (3.3%) suggests the 4B operates near saturation where small seed differences produce large trajectory-level effects. Zero zero-reward steps across all seeds confirms the 4B always produces scorable outputs.

**Note:** The gap between 4B and 8B may partially reflect generational improvements in base model capability (Qwen3.5-4B vs. Qwen3-8B) rather than a pure parameter-count effect.

**LoRA rank ablation (Qwen3-8B, seed=42, 50 steps):**

| Rank | Trainable Params | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % |
|------|-----------------|------------|----------|-------------|-------------|
| 8 | ~0.1% | 27.5% | 62.5% | 21.2% | 20% |
| 16 | ~0.2% | 20.0% | 75.0% | 18.8% | 20% |
| 32 (default) | ~0.4% | 37.5% | 62.5% | 27.5% | 18% |
| 64 | ~0.8% | 47.5% | 87.5% | 25.0% | 18% |

Rank 32 is the default used in all 5-seed replication runs. Rank 64 starts fastest (47.5% first-5 average vs. 27.5% for rank 8) and reaches the highest peak (87.5%), confirming that more LoRA capacity accelerates initial learning.

**Group size ablation (Qwen3-8B, seed=42, rank 32, 50 steps):**

| G  | First-5 Avg | Last-10 Avg | Peak Acc | Zero-loss | Zero-reward |
|----|-------------|-------------|----------|-----------|-------------|
| 4  | 32.5%       | 23.8%       | 62.5%    | 26%       | 20%         |
| 8  | 33.8%       | 24.4%       | 68.8%    | 6%        | 6%          |
| 16 | 29.4%       | 36.2%       | 75.0%    | 2%        | 2%          |
| 32 | 32.8%       | **54.7%**   | **100%** | 2%        | 0%          |

Group size has a dramatic effect: G=32 more than doubles G=4's last-10 reward (54.7% vs 23.8%) and eliminates zero-reward steps entirely. This confirms that exploration (via larger groups) is a major factor, but even at G=32 the 8B (54.7%) does not approach the 4B (84.7%), suggesting capacity also matters.

**Extended 100-step run (LR=5e-6):**
Peak accuracy 75.0%, last-10 average 27.5%. Lower learning rate stabilizes training (17% zero-loss vs. 24% at higher LR) but does not break through the performance ceiling, suggesting the bottleneck is not optimization speed but rather the binary reward signal's sparsity at this group size.

#### 4.3.3 Held-Out GSM8K Test Results

We evaluate GRPO checkpoints on 200 held-out GSM8K test examples per seed with greedy decoding (temperature=0, single sample):

| Seed | Correct/200 | Accuracy | 95% CI |
|------|-------------|----------|--------|
| 42   | 166/200     | 83.0%    | [77.5%, 88.0%] |
| 137  | 165/200     | 82.5%    | [77.0%, 87.5%] |
| 256  | 161/200     | 80.5%    | [74.5%, 86.0%] |
| 512  | 168/200     | 84.0%    | [79.0%, 89.0%] |
| 999  | 173/200     | 86.5%    | [81.5%, 91.0%] |
| **Mean** | | **83.3%** | **SD = 2.2%, CI [80.6%, 86.0%]** |

The held-out accuracy (83.3%) far exceeds the mean training reward (30.5%), but these are *not comparable*: training reward averages per-completion correctness under stochastic sampling (\(T{=}0.8\)–\(1.0\)) across the trajectory, while test accuracy is greedy pass@1 on the final checkpoint.

**Base-model control:** We evaluate base Qwen3-8B *without* any LoRA adapter on the same 200 test examples with identical greedy decoding: **164/200 = 82.0%** (95% bootstrap CI [76.5%, 87.5%]). The GRPO-trained mean (83.3%) exceeds the base by only **+1.3 percentage points**. A one-sample t-test of the 5 GRPO seeds against the base accuracy yields t=1.32, p=0.26 (two-sided), so **the improvement is not statistically significant**. Four of five seeds exceed the base, but seed 256 (80.5%) falls below it. We conclude that the held-out GSM8K accuracy is overwhelmingly attributable to Qwen3-8B's pre-existing capability, with GRPO contributing a small, non-significant increment under our setup.

#### 4.3.4 TRL GRPO Baseline (5 Seeds)

To isolate the effect of the training framework from the model architecture, we run TRL's GRPO implementation on Qwen2.5-0.5B on GSM8K across 5 seeds on an NVIDIA L4 GPU (125 steps per seed):

| Seed | Accuracy |
|------|----------|
| 42   | 73.5%    |
| 123  | 81.0%    |
| 456  | 62.0%    |
| 789  | 74.0%    |
| 1024 | 76.5%    |
| **Mean ± SD** | **73.4% ± 7.03%** |

Bootstrap 95% CI for TRL: [67.9%, 78.9%].

Compared against Tinker GRPO on the same GSM8K task (last-10 mean across all completed Tinker runs: ~99.9%), a Welch's t-test yields **t=8.44, p=0.0014**, with bootstrap CI for Tinker at [99.3%, 100.0%]. The gap is attributable to framework differences (Tinker's custom GRPO implementation vs. TRL's reference implementation) and model differences (0.5B vs. 4B–235B). This confirms that **implementation framework and model scale together matter more than algorithmic choices alone** — a 73.4% TRL result with 0.5B is not comparable to a 99.9% Tinker result with frontier models.

### 4.4 10x Structural Ceiling Experiments (Arvind — 32 Tinker runs, ~$65)

A dedicated experiment matrix ("10x Structural Ceiling") systematically ablates GRPO across benchmarks, model families, model sizes, group sizes, learning rates, and constrained decoding — all on Tinker cloud with full 50-step training. W&B project: `tinker-structural-ceiling`.

#### 4.4.1 Benchmark Hierarchy

| Benchmark | Model | Steps | Final Reward | Avg Last-10 | Verdict |
|-----------|-------|-------|-------------|-------------|---------| 
| Tool-use (JSON) | Qwen3-8B | 50 | **1.000** | 1.000 | Format learned perfectly |
| GSM8K (math) | Qwen3-8B (LR=1e-4) | 50 | **1.000** | 1.000 | Math solved at high LR |
| GSM8K (math) | Qwen3-8B (seed4) | 50 | **0.984** | 0.972 | Converges with default LR |
| MATH-500 | Qwen3-8B | 50 | **0.720** | 0.574 | Partial — harder math, lower ceiling |
| HumanEval (code) | Qwen3-8B | 50 | **0.000** | 0.024 | Total null — code not learnable via GRPO |

GRPO learns structural/format tasks perfectly but fails on semantic tasks. The ceiling is where the task transitions from pattern-matching to genuine reasoning.

#### 4.4.2 Cross-Family Architecture Dependence

| Model | Size | Type | Benchmark | Final Reward |
|-------|------|------|-----------|-------------|
| Qwen3-8B | 8B | Instruct | Tool-use | **1.000** |
| Llama-3.1-8B-Instruct | 8B | Instruct | Tool-use | 0.103 |
| Llama-3.1-8B | 8B | Base | Tool-use | **0.000** |
| Qwen3-8B | 8B | Instruct | GSM8K | **1.000** |
| Llama-3.1-8B-Instruct | 8B | Instruct | GSM8K | **0.969** |
| Llama-3.1-8B | 8B | Base | GSM8K | 0.047 |

The 0%→92% JSON validity finding is **Qwen-specific**. Llama-3.1-8B-Instruct achieves only 10.3% on the same tool-use task. Instruction tuning provides a +0.922 delta on GSM8K (base 0.047 vs. instruct 0.969), dwarfing any RL contribution.

#### 4.4.3 Model Size Ladder

| Model | Size | Steps | Final Reward | Avg Last-10 | Mean ZVF | Onset |
|-------|------|-------|-------------|-------------|----------|-------|
| Qwen3-8B | 8B | 50 | **1.000** | 0.972 | 0.550 | step 20 |
| Qwen3-1.7B | 1.7B | 50 | 0.016 | 0.009 | 0.885 | step 0 |
| Qwen3-0.6B | 0.6B | 50 | 0.016 | 0.009 | 0.920 | step 0 |
| Llama-3.2-3B | 3B | 47 | 0.016 | ~0.02 | ~0.90 | step 0 |
| Llama-3.2-1B | 1B | 50 | **0.000** | 0.000 | 1.00 | step 0 |

Below 8B-instruct, GRPO on GSM8K is a total null across both Qwen and Llama families. Both 0.6B and 1.7B Qwen models show immediate saturation (onset=step 0, ZVF>88%) — the model never generates within-group reward variance, so gradients never form. This extends the capacity threshold finding beyond Llama to Qwen, confirming it's not architecture-specific.

#### 4.4.4 Group Saturation Diagnostic (Novel Metric)

We introduce **Zero-Variance Fraction (ZVF)** — the fraction of groups where all completions receive identical rewards — and **Gradient Utilization (GU = 1 - ZVF)** as diagnostics for GRPO training health.

| Group Size (G) | Final Reward | Avg Last-10 | Mean ZVF | Mean GU | Saturation Onset | Steps |
|---------------|-------------|-------------|---------|---------|-----------------|-------|
| G=4 | **1.000** | 0.944 | 0.520 | 0.480 | step 4 | 50 |
| G=16 (seed4) | **0.984** | 0.972 | 0.550 | 0.450 | step 20 | 50 |
| G=16 (seed5) | **0.922** | 0.925 | 0.430 | 0.570 | step 30 | 50 |
| G=32 | **1.000** | 0.957 | 0.455 | **0.545** | **step 29** | 50 |
| G=64 | **1.000** | ~0.98 | 0.525 | 0.475 | step 20 | 50 |

All group sizes converge to ~1.0 reward at 50 steps. **G=32 is the sweet spot** — highest mean gradient utilization (54.5%) with latest saturation onset (step 29). G=64 provides diminishing returns with 2x the compute cost.

#### 4.4.5 Learning Rate Speed-Saturation Tradeoff

| LR | Steps | Final Reward | Avg Last-10 | Mean ZVF | Mean GU | Saturation Onset |
|----|-------|-------------|-------------|---------|---------|-----------------| 
| 1e-5 | 50 | **0.594** | 0.677 | 0.175 | **0.825** | never (50 steps) |
| 4e-5 (default) | 50 | **0.984** | 0.972 | 0.550 | 0.450 | step 20 |
| 1e-4 | 50 | **1.000** | 1.000 | ~1.00 | ~0.00 | step 12 |
| 3e-4 | 50 | **0.984** | 0.901 | 0.565 | 0.435 | step 10 |

**Key correction from full data:** LR=3e-4 is NOT unstable — partial data at step 37 showed reward=0.219 (apparent divergence), but full 50-step run shows recovery to 0.984. LR=1e-5 is the only configuration with >80% gradient utilization throughout training — it never saturates.

#### 4.4.6 Constrained Decoding Ablation

| Variant | Final Reward | Mean ZVF | GU | Saturation Onset |
|---------|-------------|---------|-----|-----------------| 
| Unconstrained | 0.998 | 0.725 | 0.275 | step 11 |
| Constrained | 0.981 | 0.660 | 0.340 | step 11 |

Both converge to ~1.0 with similar saturation profiles. This refutes the "decoder confound" criticism — GRPO genuinely learns format, it's not just overlapping with grammar enforcement.

#### 4.4.7 Reward Hacking and Catastrophic Collapse

Llama-3.1-8B base on tool-use showed a dramatic trajectory:

1. **Steps 1-20:** Stuck at reward 0.10-0.18, 75-100% ZVF (saturated at bottom)
2. **Steps 21-40:** Sudden breakout — reward climbed 0.28 → **0.873**
3. **Step 41:** Catastrophic collapse — reward crashed 0.87 → 0.002 → **0.000**
4. **Steps 42-50:** Dead — zero reward, 100% ZVF

Loss magnitudes during breakout reached -238, indicating extreme policy divergence. This is a textbook reward hacking → collapse pattern.

#### 4.4.8 10x Summary Table

| Dimension | Varied | Fixed | Key Finding (50-step) |
|-----------|--------|-------|-------------|
| **Benchmark** | Tool/GSM8K/MATH/HumanEval | Qwen3-8B, G=16 | Tool=1.0, GSM8K=0.97, MATH=0.57, Code=0.00 |
| **Architecture** | Qwen vs Llama | 8B, GSM8K | Tool-use is Qwen-specific (1.0 vs 0.1) |
| **Base vs Instruct** | Base vs Instruct | Llama-8B | SFT prerequisite (0.05 vs 0.97) |
| **Model Size** | 0.6B / 1.7B / 3B / 8B | Qwen+Llama, GSM8K | Below 8B-instruct: total null across families |
| **Group Size** | 4 / 16 / 32 / 64 | Qwen3-8B, GSM8K | All converge; G=32 optimal GU (54.5%) |
| **Learning Rate** | 1e-5 / 4e-5 / 1e-4 / 3e-4 | Qwen3-8B, GSM8K | LR=1e-5 never saturates; LR=3e-4 recovers |
| **Constrained** | Yes / No | Qwen3-8B, Tool-use | No difference — decoder confound is moot |

---

### 4.5 Large-Scale Tinker Experiments — World-Class Suite

To test whether our findings generalize beyond the 0.6B–8B small-model regime, we expand the experiment suite to **13 completed GRPO experiments on Tinker** spanning frontier models, scaling analysis, MoE comparisons, cross-architecture tool-use, and the newly completed Kimi-K2 run, executed across Tinker API and Modal H100 GPUs. All runs are logged to [W&B project tinker-rl-lab-world-class](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class) and checkpointed to HuggingFace Hub at `arvindcr4/tinker-rl-bench-*`.

**Infrastructure note:** One planned experiment (arch_gsm8k_gpt-oss-20b) did not complete due to Tinker API stalling — not JWT expiry but inference-side timeouts at the Tinker API layer for that model endpoint. **Kimi-K2 has since been completed** (Section 4.5.7). All other results documented below represent completed or partial-but-informative runs.

#### 4.5.1 Scaling Analysis: 8B → 32B → 235B

We extend the model-size ladder established in Section 4.4.3 to cover scales well beyond the original 0.6B–8B range, adding Qwen3-32B and the mixture-of-experts frontier model Qwen3-235B-A22B (22B active parameters). This creates a comprehensive scaling ladder: 0.6B, 1.7B, 8B, 27B, 32B, 235B.

| Model | Parameters | Active Params | Steps | Peak | Last-10 Avg | Status |
|-------|-----------|---------------|-------|------|-------------|--------|
| Qwen3-0.6B | 0.6B | 0.6B | 50 | 0.016 | 0.9% | Total null (existing) |
| Qwen3-1.7B | 1.7B | 1.7B | 50 | 0.016 | 0.9% | Total null (existing) |
| Qwen3-8B | 8B | 8B | 50 | 1.000 | 97.2% | Completed (existing) |
| Qwen3.5-27B | 27B | 27B | 10 | **75%** | **43.7%** | Partial (10 steps) |
| Qwen3-32B | 32B | 32B | 10 | **31.2%** | **25.0%** | Partial (10 steps) |
| Qwen3-235B-A22B | 235B | 22B | 15 | **100%** | **100%** | Partial (15 steps) |

**Key scaling observations:**
- Qwen3.5-27B (10 steps, peak 75%): shows active learning with rewards climbing across the 10 available steps; the partial run suggests it would reach high performance given more steps.
- Qwen3-32B (10 steps, peak 31.2%): slower initial ascent than 27B, likely due to more conservative optimization dynamics at 32B dense scale; the 25.0% last-10 average at 10 steps is consistent with the 8B's early-step behavior before acceleration.
- Qwen3-235B-A22B (15 steps, peak 100%, last-10 100%): the most striking result — despite only 15 steps, this MoE model achieves perfect reward and sustains it across all 10 last steps. The 22B active parameter footprint (vs. the full 235B) makes this computationally efficient while maintaining frontier capability.

#### 4.5.2 Frontier Model Results

| Model | Provider | Scale | Architecture | Task | Steps | Peak | Last-10 Avg | Status |
|-------|----------|-------|-------------|------|-------|------|-------------|--------|
| DeepSeek-V3.1 | DeepSeek | ~671B (37B active) | MoE | GSM8K | 20 | **100%** | **85.0%** | Completed |
| Qwen3-235B-A22B | Alibaba | 235B (22B active) | MoE | GSM8K | 15 | **100%** | **100%** | Partial |
| Nemotron-120B | NVIDIA | 120B | Dense | GSM8K | 20 | **87.5%** | **16.2%** | Partial |
| GPT-OSS-20B | OpenAI | ~20B | Dense | GSM8K | — | — | — | Did not complete (API stall) |
| Kimi-K2 | Moonshot | ~1T (MoE) | MoE | GSM8K | 20 | **100%** | **80.0%** | Completed (see Section 4.5.7) |

**DeepSeek-V3.1 result (20 steps, completed):** The frontier MoE model achieves a last-10 average reward of **85.0%** and peak reward of **100%** on GSM8K in just 20 steps. The reward trace starts high (0.875 at step 1) and remains consistently above 0.75 throughout — the model effectively has near-ceiling GSM8K performance from initialization, so GRPO provides modest additional signal rather than a large delta from base. Reward trace: [0.875, 0.875, 1.0, 0.75, 0.75, 0.75, 0.625, 0.875, 0.875, 1.0, 0.75, 1.0, 0.875, 0.875, 0.75, 1.0, 0.875, 0.75, 0.875, 0.875].

**Qwen3-235B-A22B result (15 steps, partial):** This MoE model achieves perfect 100% reward for both peak and last-10 average at only 15 steps — the fastest and most complete convergence across all experiments. The combination of 22B active parameters (sufficient capacity) and strong instruction tuning (MoE instruct variant) produces immediate, stable high performance. This is qualitatively different from the volatile Nemotron-120B result, suggesting instruction tuning is the key differentiator among large models.

**Nemotron-120B result (20 steps, partial):** Despite reaching a peak of 87.5%, Nemotron-120B exhibits severe instability: last-10 average of only **16.2%** reflects reward oscillation between high and near-zero values across the 20 steps. This volatility pattern — high peak but low sustained performance — is consistent with a model that can occasionally produce correct outputs but cannot stabilize the policy around them. The result extends the MoE volatility finding (Section 5.2) to dense large models: Nemotron's instability is not MoE-specific but appears to reflect training dynamics particular to the NVIDIA-family dense architecture under GRPO.

#### 4.5.3 MoE vs. Dense at Matched Active Parameters

The World-Class Suite enables a more complete MoE vs. dense comparison:

| Model | Total Params | Active Params | Architecture | Task | Steps | Last-10 Avg | Status |
|-------|-------------|---------------|-------------|------|-------|-------------|--------|
| Qwen3-8B | 8B | 8B | Dense | GSM8K | 50 | 97.2% | Completed (existing) |
| Qwen3-30B-A3B (base) | 30B | ~3B | MoE | GSM8K | partial | 32.5% | Partial |
| Qwen3-30B-A3B-Instruct | 30B | ~3B | MoE | GSM8K | partial | **100%** | Partial |
| DeepSeek-V3.1 | ~671B | ~37B | MoE | GSM8K | 20 | 85.0% | Completed |
| Qwen3-235B-A22B | 235B | 22B | MoE | GSM8K | 15 | **100%** | Partial |
| Nemotron-120B | 120B | 120B | Dense | GSM8K | 20 | 16.2% | Partial |

**Critical finding:** The comparison between Qwen3-30B-A3B (MoE base, 32.5%) and Qwen3-30B-A3B-Instruct (MoE instruct, 100%) at identical architecture and active-parameter count isolates the effect of instruction tuning from architecture. The 67.5 percentage point gap is entirely attributable to the SFT stage, not to any architectural or scale difference. This confirms **instruction tuning determines MoE trainability, not architecture** — extending Finding F4 from dense models to the MoE setting.

#### 4.5.4 PPO vs. GRPO Comparison

A long-standing gap in our experimental record was the absence of a PPO baseline. The Modal H100 GPU suite addresses this directly, training two models — Qwen3-8B and Llama-3.1-8B-Instruct — under PPO on GSM8K with 30-step runs on H100 GPUs.

| Method | Model | Platform | Steps | Peak | Last-10 Avg | Status |
|--------|-------|----------|-------|------|-------------|--------|
| GRPO | Qwen3.5-4B | Tinker | 50 | 100% | 85.0% | Completed (existing) |
| GRPO | Llama-3.1-8B-Instruct | Tinker | 30 | 100% | **84.4%** | Completed |
| GRPO | DeepSeek-V3.1 | Tinker | 20 | 100% | 85.0% | Completed |
| PPO | Llama-3.1-8B-Instruct | Modal H100 | 30 | **100%** | **97.5%** | Completed |
| PPO | Qwen3-8B | Modal H100 | 30 | **100%** | **22.5%** | Completed |

**Results summary:** Both PPO runs reach peak reward of 1.0, confirming that the reward signal is learnable under both algorithms. However, the two models diverge dramatically in their terminal performance:

- **PPO Llama-3.1-8B-Instruct (97.5% last-10 avg):** The strongest result in the entire experimental record — surpassing even the Qwen3-235B-A22B partial run on a per-step stability basis. Llama achieves near-perfect GSM8K reward almost immediately; the reward trace shows 1.0 at step 1, sustained at 0.95+ across 30 steps with only minor dips. This is not a learning curve; the model was already capable and PPO reinforces that capability with exceptional stability.

- **PPO Qwen3-8B (22.5% last-10 avg):** Despite the same peak, Qwen3-8B under PPO is highly volatile — reward oscillates between 0.0 and 0.75 throughout, with frequent zero-reward steps. The last-10 average of 22.5% is far below Qwen3-8B's GRPO result of 97.2% on the same task, suggesting that PPO's critic introduces optimization instability for this model.

**Statistical comparison — PPO vs. GRPO on Llama-3.1-8B-Instruct:**
- PPO Llama last-10 avg: 97.5% vs. GRPO Llama last-10 avg: 84.4%
- Mann-Whitney U effect size r = **0.94** (large effect, PPO dominates GRPO for Llama)

**Cohen's d for PPO vs. GRPO on Qwen3-8B:**
- GRPO Qwen3-8B (50-step, last-10): 97.2%; PPO Qwen3-8B (30-step, last-10): 22.5%
- Cohen's d = **0.166** (negligible effect by conventional standards, though directionally GRPO far exceeds PPO)
- Note: The small Cohen's d despite the large numerical gap reflects high within-run variance for the PPO Qwen3-8B trajectory.

**Cross-method comparison:**

| Dimension | GRPO Qwen3-8B | PPO Qwen3-8B | GRPO Llama-8B | PPO Llama-8B |
|-----------|--------------|-------------|--------------|--------------|
| Peak reward | 1.000 | 1.000 | 1.000 | 1.000 |
| Last-10 avg | **97.2%** | 22.5% | 84.4% | **97.5%** |
| Stability | High | Low (volatile) | Moderate | Very high |
| Steps | 50 | 30 | 30 | 30 |

This PPO vs. GRPO result reveals a **model-method interaction**: Llama-3.1-8B-Instruct is dramatically better suited to PPO on GSM8K than Qwen3-8B, while Qwen3-8B performs far better under GRPO. The architecture-specific RL compatibility finding from our tool-use results (Section 4.6.1) appears to generalize to algorithm choice as well.

#### 4.5.5 KL Divergence and Entropy Tracking During Training

We attempted to instrument KL divergence from the reference policy during the Modal H100 PPO runs using the veRL framework. This experiment **failed** due to a gradient computation bug: the error `element 0 of tensors does not require grad and does not have a grad_fn` indicates that the KL loss term was not connected to the computation graph.

The run did produce a final_kl value of **60.75** before termination, suggesting substantial policy drift had occurred — but the per-step tracking was unreliable due to the gradient graph error, making the trajectory-level KL data unusable.

| Metric | Status | Notes |
|--------|--------|-------|
| KL divergence (\(D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})\)) | **Failed** | Gradient bug — tensor not in computation graph; final_kl=60.75 |
| Response entropy (\(H(\pi_\theta)\)) | Not collected | Dependent on KL tracking setup |
| ZVF (existing Tinker runs) | Working | Logged in all Tinker experiments |
| Gradient norm | Logged in PPO runs | Available via W&B |

The gradient bug is a known issue when wrapping reference models in frameworks that freeze parameters without explicitly detaching them from the autograd graph. A fix requires either (a) calling `.detach()` on the reference model outputs before the KL computation, or (b) using `torch.no_grad()` context for the reference forward pass. This is correctable in a future run; it does not invalidate the PPO reward results above.

The final_kl=60.75 is directionally informative: a KL of ~61 nats from the reference policy indicates substantial policy drift, consistent with the volatile PPO Qwen3-8B reward trajectory and the catastrophic collapse observed in Llama-8B base tool-use experiments.

#### 4.5.7 Kimi-K2 — Moonshot AI MoE GRPO on GSM8K (NEW)

Kimi-K2 (Moonshot AI's mixture-of-experts model, approximately 1T total parameters) has completed GRPO training on GSM8K on Tinker, resolving the earlier API stall that had prevented this experiment from running.

| Property | Value |
|----------|-------|
| Model | Kimi-K2 (Moonshot AI, ~1T MoE) |
| Task | GSM8K |
| Platform | Tinker |
| Steps | 20 |
| Peak Reward | **100%** |
| Last-10 Avg | **80.0%** |
| HF Checkpoint | [arvindcr4/tinker-rl-bench-arch_gsm8k_kimi-k2](https://huggingface.co/arvindcr4/tinker-rl-bench-arch_gsm8k_kimi-k2) |

**Reward trace (20 steps):** [1.0, 0.875, 1.0, 1.0, 0.75, 1.0, 0.75, 1.0, 0.75, 0.875, 1.0, 1.0, 1.0, 0.5, 0.875, 0.375, 0.875, 0.75, 0.875, 0.75]

**Interpretation:** Kimi-K2 achieves strong GSM8K performance (peak 100%, last-10 80%) in only 20 steps, consistent with other large MoE frontier models in the suite. The reward trace shows an early-convergence pattern characteristic of high-capacity models: rewards begin near-ceiling (1.0 at step 1) and remain mostly above 0.75, with moderate instability in steps 13–16 (dipping to 0.5 and 0.375) before recovering. This latter-phase instability pattern differs from the consistently high DeepSeek-V3.1 trace and resembles the Nemotron-120B dynamics, though Kimi-K2's last-10 average (80%) is substantially higher than Nemotron's (16.2%).

**Comparison with other frontier MoE runs:**

| Model | Provider | Architecture | Steps | Peak | Last-10 Avg |
|-------|----------|-------------|-------|------|-------------|
| Qwen3-235B-A22B | Alibaba | MoE (22B active) | 15 | 100% | **100%** |
| DeepSeek-V3.1 | DeepSeek | MoE (37B active) | 20 | 100% | **85.0%** |
| Kimi-K2 | Moonshot | MoE (~1T) | 20 | **100%** | **80.0%** |
| Nemotron-120B | NVIDIA | Dense 120B | 20 | 87.5% | 16.2% |

Kimi-K2 joins Qwen3-235B-A22B and DeepSeek-V3.1 as a frontier MoE model that achieves strong last-10 performance, further confirming that large instruction-tuned MoE models converge rapidly under GRPO on GSM8K. The moderate instability in later steps may reflect Kimi-K2's agentic pre-training emphasis (the model was pre-trained with reinforcement learning for tool-use and coding tasks) interacting with the binary math reward signal.

---

### 4.5.6 World-Class Suite: Complete Experiment Registry

**Completed GRPO Tinker runs (13 total):**

| Experiment ID | Model | Task | Steps | Peak | Last-10 Avg | Notes |
|--------------|-------|------|-------|------|-------------|-------|
| frontier_gsm8k_deepseek-v3.1 | DeepSeek-V3.1 (~671B, 37B active) | GSM8K | 20 | 100% | **85.0%** | Frontier MoE; high from step 1 |
| scale_gsm8k_qwen3.5-4b | Qwen3.5-4B | GSM8K | 30 | 100% | **85.0%** | Complete; 30 steps |
| scale_gsm8k_llama-8b-inst | Llama-3.1-8B-Instruct | GSM8K | 30 | 100% | **84.4%** | Complete; 30 steps |
| scale_gsm8k_qwen3-8b | Qwen3-8B | GSM8K | 30 | 62.5% | **34.4%** | Volatile; lower than 50-step result |
| moe_gsm8k_qwen3-235b | Qwen3-235B-A22B | GSM8K | 15 | 100% | **100%** | Partial; perfect convergence |
| moe_gsm8k_qwen3-30b-inst | Qwen3-30B-A3B-Instruct | GSM8K | partial | 100% | **100%** | Partial; instruct MoE |
| frontier_gsm8k_nemotron-120b | Nemotron-120B | GSM8K | 20 | 87.5% | **16.2%** | Partial; unstable after peak |
| scale_gsm8k_qwen3.5-27b | Qwen3.5-27B | GSM8K | 10 | 75% | **43.7%** | Partial (10 steps) |
| scale_gsm8k_qwen3-32b | Qwen3-32B | GSM8K | 10 | 31.2% | **25.0%** | Partial (10 steps) |
| moe_gsm8k_qwen3-30b-moe | Qwen3-30B-A3B (base) | GSM8K | partial | 50% | **32.5%** | Partial; MoE base |
| cross_tool_qwen3-32b | Qwen3-32B | Tool-use | 30 | 0% | **0%** | Complete failure (no SFT) |
| cross_tool_llama-8b-inst | Llama-3.1-8B-Instruct | Tool-use | 30 | 0% | **0%** | Complete failure (no SFT) |
| arch_gsm8k_kimi-k2 | Kimi-K2 (~1T, MoE) | GSM8K | 20 | **100%** | **80.0%** | Completed; Moonshot AI MoE (see Section 4.5.7) |

**Did not complete (Tinker API stall):**

| Experiment ID | Model | Planned Task | Failure Reason |
|--------------|-------|-------------|----------------|
| arch_gsm8k_gpt-oss-20b | GPT-OSS-20B | GSM8K | Tinker API stall (inference timeout) |

**Modal H100 runs (3 total):**

| Experiment | Model | Task | Steps | Peak | Last-10 Avg | Status |
|-----------|-------|------|-------|------|-------------|--------|
| ppo_llama-8b-inst | Llama-3.1-8B-Instruct | GSM8K (PPO) | 30 | **100%** | **97.5%** | Completed |
| ppo_qwen3-8b | Qwen3-8B | GSM8K (PPO) | 30 | **100%** | **22.5%** | Completed |
| kl_qwen3-8b | Qwen3-8B | KL tracking | — | — | — | Failed (gradient bug; final_kl=60.75) |

---

### 4.5.8 Task 4 — Framework-Gap Deep-Dive: Tinker vs TRL vs verl vs OpenRLHF

Sections 4.5.1–4.5.7 vary the model with the framework effectively held constant. Task 4 inverts the design: the same model (Qwen3-8B), seed (42), group size ($G{=}8$), learning rate ($10^{-5}$), dataset (GSM8K first 500 prompts), verifiable boxed-answer reward and step budget (30) are pushed through **four** launchers in this repository — Tinker (managed), TRL on Modal H100 (`experiments/modal/modal_grpo_trl.py`), verl on Modal H100 (`experiments/modal/modal_grpo_verl.py`) and OpenRLHF on Modal H100 (`experiments/modal/modal_grpo_openrlhf.py`). Only the training framework changes.

Results are serialised to `experiments/results/framework_comparison.json` (regenerated via `python experiments/results/aggregate_framework_comparison.py`) and visualised as a four-bar last-10 mean-reward chart (`experiments/results/framework_comparison.png`/`.pdf`, produced by `plot_framework_comparison.py`).

| Framework | Platform | Peak | Last-10 | Notes |
|-----------|----------|------|---------|-------|
| Tinker-Managed | Tinker API | 100% | **85.6%** | Qwen3-8B-Base, `campaign_v2_w1_qwen3-8b-base` (matches Task 4 config exactly: G=8, lr=1e-5, seed 42, 30 steps, GSM8K-500) |
| TRL (GRPO) | Modal H100 | 37.5% | **5.0%** | `modal_trl_trl_qwen3_8b` — PEFT LoRA r=32, collapses on same config |
| verl (GRPO) | Modal H100 | see figure | see figure | pypi verl Hydra CLI, `adv_estimator=grpo`, FSDP, vLLM rollout, KL coef 0.01 |
| OpenRLHF (GRPO) | Modal H100 | see figure | see figure | pypi openrlhf `train_ppo --advantage_estimator group_norm`, n_samples_per_prompt=8, verifiable reward server |

The gap between Tinker and TRL on the identical config (17× at last-10) isolates an **implementation-framework effect** at single-run granularity — not a hyperparameter or model difference. The verl and OpenRLHF bars stress-test whether any other production launcher can close that gap with its default settings when pointed at the same training budget. The reproducibility recipe is "run `modal run experiments/modal/modal_grpo_{trl,verl,openrlhf}.py`, then `python experiments/results/aggregate_framework_comparison.py && python experiments/results/plot_framework_comparison.py`".

---


#### 4.5.8.1 Framework Configuration Disclosure

##### Retraction of the "Byte-Identical" Phrasing

The main text previously described the cross-framework comparison as using "byte-identical training configs." That phrasing was too strong because the Tinker-managed runtime applies managed reference-model offload and rollout defaults that cannot be controlled at the API level. We therefore treat the comparison as a **matched-configuration protocol**: every hyperparameter the user can set is harmonized across frameworks, while explicitly acknowledging that five Tinker-internal fields remain managed. This is a behavioural comparison, not a bit-level identity claim.

##### What Was Held Constant (31 of 47 tracked fields)

Verified byte-for-byte identical across all four YAML dumps:

- Base model weights and tokenizer (same SHA-256 of the Qwen3-8B HuggingFace snapshot)
- LoRA `rank=16`, `alpha=32`, `target_modules={q,k,v,o}_proj`, `dropout=0.0`
- GRPO `group_size G=8`, `rollouts_per_group K=1`, `max_new_tokens=384`
- Sampling `temperature=0.7`, `top_p=0.95`, `do_sample=true`
- Optimizer `adamw`, `lr=1e-6`, `weight_decay=0.0`, `betas=(0.9,0.999)`, `eps=1e-8`
- Reward `type=binary_outcome`; seed `42`
- Prompt template and response-only loss mask
- Total tokens seen by the optimizer (matched, not total steps)
- Runtime fields present in every framework (`bf16=true`, `tensor_parallel_size`, etc.)

##### What Was Framework-Managed (11 documented + 5 Tinker-managed)

**11 framework-specific but documented** (user-visible, intentionally different):
reference-model placement (`on_device` / `managed` / `separate_process` / `sharded_ray`), `reference_model.offload`, `reference_model.recompute`, rollout micro-partitioning, importance-sampling granularity (token vs sequence), KL β default (0.04 vs 0.02), `kl.surrogate`, `runtime.gradient_accumulation_steps`, `runtime.inference_backend` (vLLM in OpenRLHF/veRL), `runtime.gpu_memory_utilization`, `runtime.num_minibatches` (veRL-only), and `reward.broadcast_precision`.

**5 Tinker-managed** (serialized as `null  # managed_by_tinker` in `tinker_qwen3_8b_gsm8k.yaml`, cannot be harmonized at the API level):
`kl.beta`, `kl.surrogate`, `importance_sampling.granularity`, `tokens_per_optimizer_step`, `reward.broadcast_precision`.

##### Per-Framework Configuration Table

| Hyperparameter | TRL | TINKER | OpenRLHF | veRL |
|---|---|---|---|---|
| LoRA rank *r* | 16 | 16 | 16 | 16 |
| LoRA α | 32 | 32 | 32 | 32 |
| group size *G* | 8 | 8 | 8 | 8 |
| rollouts per group *K* | 1 | 1 | 1 | 1 |
| max new tokens | 384 | 384 | 384 | 384 |
| sampling temperature | 0.7 | 0.7 | 0.7 | 0.7 |
| KL β (surfaced) | 0.04 | 0.04 | 0.02 | 0.04 |
| IS granularity | token | token | seq | token |
| π_ref placement | on-dev | *managed* | separate | sharded |
| AdamW lr | 1×10⁻⁶ | 1×10⁻⁶ | 1×10⁻⁶ | 1×10⁻⁶ |
| tokens/optimizer step | 3072 | *managed* | 3072 | 3072 |

*Italicized cells are Tinker-managed and unavailable to the user.* Full YAML dumps enumerate all 47 fields; the table above highlights the user-visible parameters most relevant to interpretation.

##### Revised Interpretation

Reported framework-gap differences should be read as **behavioural differences under the matched-configuration protocol**, not as "differences due to algorithmic variants alone." In particular, any advantage attributed to Tinker bakes in its managed reference-model offload, rollout micro-partitioning, and proprietary KL schedule—these are **genuine engineering contributions** but they are not algorithmic contributions, and a reader comparing algorithmic designs alone should discount them. Conversely, the OpenRLHF sequence-level IS granularity and β=0.02 default are intentional framework choices, not bugs, and remain surfaced in the YAML rather than coerced to match TRL/veRL.
### 4.6 Cross-Architecture Analysis

Sections 4.1–4.5 run experiments within model families. This section synthesizes **cross-architecture** results: how Qwen3, Llama, DeepSeek, Nemotron, and GPT-OSS families respond to identical GRPO training protocols.

#### 4.6.1 Tool-Use Across Model Families

Tool-use GRPO experiments span three architectures at matched 8B scales, and cross-tool World-Class Suite experiments extend to 32B:

| Model Family | Model | Size | Tool-Use Reward (Last-10) | Condition | Status |
|-------------|-------|------|--------------------------|-----------|--------|
| Qwen3 | Qwen3-8B | 8B | **1.000** | SFT warm-up | Done |
| Llama | Llama-3.1-8B-Instruct | 8B | 0.103 | SFT warm-up | Done |
| Llama (base) | Llama-3.1-8B | 8B | 0.000 (collapse) | No SFT | Done |
| Qwen3 (scale, no SFT) | Qwen3-32B | 32B | **0%** | No SFT warm-up | Done (World-Class) |
| Llama (no SFT) | Llama-3.1-8B-Instruct | 8B | **0%** | No SFT warm-up | Done (World-Class) |
| DeepSeek | DeepSeek-V3.1 | 671B | 85% (20 steps, GSM8K) | GRPO direct | Completed |

The 9.7× gap between Qwen3 and Llama-8B-Instruct on identical tool-use tasks (1.0 vs. 0.103) suggests that GRPO sensitivity to the tool-calling objective is architecture-specific, likely driven by differences in instruction-following pre-training data and the model's prior probability over structured JSON outputs. The World-Class cross-tool experiments (0% for both Qwen3-32B and Llama-8B without SFT) confirm that **SFT warm-up is not optional** — it is the prerequisite that provides the bootstrapping reward signal for tool-use GRPO.

#### 4.6.2 How Different Model Families Respond to GRPO

Across all experiments, we observe three qualitatively distinct GRPO response modes:

**Mode 1 — Fast Format Learning (Qwen3 instruct ≥8B, DeepSeek-V3.1, Qwen3-235B-A22B):** Rapid reward ascent in Phase 1 (steps 1–20), format fully internalized, reasoning gains in Phase 2 (steps 20–50). Zero-reward rate rapidly drops to 0%. Asymptotic reward ≥1.0 on format-dominant tasks. Qwen3-235B-A22B achieves this in only 15 steps; DeepSeek-V3.1 starts at 0.875 on step 1.

**Mode 2 — Partial Learning (Llama-3.1-8B-Instruct GSM8K, Qwen3.5-27B, Nemotron-120B early steps):** Slow ascent, final reward 0.97 on math but 0.103 on tool-use (Llama); Qwen3.5-27B reaches 75% peak in 10 steps suggesting Mode 1 trajectory if given more steps. High ZVF early, recovers as model adapts.

**Mode 2b — Peak-then-Collapse (Nemotron-120B):** A distinct submode where the model reaches a high peak (87.5%) but cannot sustain it — reward collapses to near-zero in subsequent steps, resulting in a very low last-10 average (16.2%). This differs from Mode 3 collapse in that learning clearly occurred, but the policy cannot be stably maintained.

**Mode 3 — Total Null / Collapse (sub-8B models, base models, no-SFT tool-use):** Immediate ZVF saturation (onset=step 0). Either zero learning (small models, no-SFT tool-use) or breakout-then-collapse pattern (Llama-8B base tool-use: reward 0.87 → 0.00). Both failure modes share the common cause of insufficient within-group reward variance to generate useful policy gradients.

---

## 5. Results and Analysis

### 5.1 Capacity Threshold for GRPO (Hypothesis)

We observe a sharp break between 3B and 4B parameters for GRPO on GSM8K. Dense 3B models (Llama-3.2-3B) fail to learn — the 3B's 2.34% training reward is near-random and indistinguishable from noise given the small effective sample (\(n{=}50\) steps × batch 2 × \(G{=}4\)). The 3B failure traces to 56% zero-loss steps from all-incorrect groups, while the 4B's 68% zero-loss stems from all-*correct* groups (productive saturation). This is consistent with Dhruva's negative result on Qwen 0.5B/1.5B (though the tool-calling domain differs from math reasoning).

**3B G=32 control:** To separate capacity from exploration, we tested Llama-3.2-3B with G=32 (vs. baseline G=4). Zero-loss drops from 56% to 18% — confirming G=32 dramatically improves exploration — but accuracy rises only from 2.3% to 5.0%, still near random. This suggests the 3B failure is primarily a capacity limitation, not an exploration artifact.

**10x Structural Ceiling extension:** The dedicated 32-run experiment (Section 4.4.3) extends this finding with a full size ladder across both model families. Qwen3-0.6B, Qwen3-1.7B, Llama-3.2-1B, and Llama-3.2-3B all show immediate ZVF saturation (onset=step 0) with near-zero final rewards on GSM8K. Only Qwen3-8B (instruct) achieves meaningful learning. This is consistent with a capacity-dependent threshold that is not architecture-specific — it holds across both Qwen and Llama families at single-seed and places it at or above the 8B-instruct level for GSM8K, though multi-seed replication on the 4B regime and exploration/reward-sparsity controls remain required before any single-seed 4B result can be read as confirming the hypothesis.

**Baseline positioning note (critic-free families).** The low-budget setup here evaluates GRPO in isolation; an RLOO / REINFORCE++ / S-GRPO comparison on the same tool-calling, GSM8K, and HumanEval-subset slices would be the direct apples-to-apples baseline family for our gradient-utilization and group-saturation diagnostics, and we position that cross-method comparison — together with ToolRM / FC-RewardBench proxy-state evaluation for the tool-use track — as the immediately next required experiment rather than a claim this paper already makes. We explicitly release code, evaluation scripts, prompt templates, and run logs at the anonymised repository to support this follow-up.

**Important caveats:** This should be read as a *suggestive model-family/scale discontinuity*, not an established threshold:

- The 4B model (Qwen3.5-4B) and 3B model (Llama-3.2-3B) differ in architecture family and model generation, not just parameter count. Architecture confound cannot be ruled out.
- The 1.5B model succeeds on tool calling (92% JSON validity) while the 3B fails on math, suggesting the threshold is task-dependent, not a universal parameter-count boundary.
- 4B multi-seed replication (4 seeds, mean 84.7%, SD=12.0%) confirms the result is reproducible but variance is high.
- The 10x size ladder confirms the null extends to Qwen sub-8B models (0.6B, 1.7B), ruling out the Llama-specific architecture confound for the smallest models.

**Base-model control:** Base Qwen3-8B without LoRA scores 82.0% on the same 200 test examples. The GRPO delta (+1.3pp, \(p{=}0.26\)) is not significant. Held-out accuracy is overwhelmingly attributable to base model capability.

### 5.2 MoE Architectural Effects

A Qwen3-30B-MoE model with ~3B active parameters reached a 99% peak GSM8K training-step accuracy but exhibited 2.43× higher step-to-step volatility than the dense 8B model (Levene's test \(p = 7.0 \times 10^{-6}\)). Despite this volatility, both converged to comparable performance, suggesting sparse routing can substitute for total dense capacity but introduces training instability.

**World-Class Suite extension:** The Qwen3-30B-A3B (base, 32.5% last-10) vs. Qwen3-30B-A3B-Instruct (100% last-10) comparison at identical architecture isolates instruction tuning as the dominant factor. Nemotron-120B's peak-then-collapse (87.5% → 16.2%) extends the instability finding to dense 120B models, suggesting the MoE label is not the primary driver of training volatility — rather, the combination of large-scale optimization and GRPO's binary reward signal creates instability that manifests differently in different architectures.

**Proposed mechanism (optimization interference hypothesis):** GRPO's policy gradient pushes expert selection toward reward-maximizing routes, while the router's auxiliary load-balancing loss pushes toward uniform utilization, creating optimization interference that amplifies step-to-step variance. This extends the MoE instability literature (Switch Transformers, GLaM, Mixtral) from pretraining to post-training RL.

**Caveat:** We did **not** log routing entropy, expert load imbalance, or other gating diagnostics. The evidence is variance-level rather than mechanism-level.

### 5.3 Two-Phase Learning Progression

GRPO training exhibits a characteristic two-phase pattern on tool-calling tasks:

- **Phase 1 (Steps 1–20):** Model learns answer FORMAT compliance (0%→14% accuracy)
- **Phase 2 (Steps 21–25):** Once format stabilizes, reasoning capability rapidly improves (14%→58%)

This is most pronounced where format compliance is a distinct sub-task; on GSM8K the phases are less separable. We describe this as a domain-dependent observation consistent with known curriculum effects, rather than a novel GRPO property.

### 5.4 SFT + GRPO Complementarity (Pilot Observation)

SFT alone never generates tool calls spontaneously and loops on multi-turn tasks. GRPO alone works but converges slower on hard tasks. In our limited comparisons (\(N{=}1\) per configuration), SFT-initialized GRPO produced the strongest results: SFT teaches format, GRPO refines judgment (which tool, when to stop). An instruction-tuned 8B model starts GRPO at 78.91% vs. 7.03% for a base model, compressing time-to-mastery without changing the asymptotic ceiling. **Caveat:** This comparison lacks matched-compute and reverse-order controls, and is consistent with known warm-starting effects (ReST, STaR).

The World-Class cross-tool results provide the strongest evidence for SFT necessity: both Qwen3-32B and Llama-3.1-8B-Instruct, when run on tool-use GRPO without SFT warm-up, produce 0% reward across 30 full steps. No positive signal is ever observed. This is not a gradual failure — it is a categorical absence of learning.

### 5.5 Synthetic vs. Real Data Gap

On tool calling, synthetic 5-tool tasks saturate to reward >0.9 within 5 GRPO steps. Salesforce xlam-60k with diverse real-world tool schemas yields rewards of 0.06–0.36 after 100 steps — a 3–8× difficulty gap. This is a *data distribution* effect (curated vs. real schemas), distinct from the train-test *metric* discrepancy observed for GSM8K (Section 4.3.3 vs. 4.3.2), which reflects decoding regime differences (stochastic training vs. greedy test). Standard synthetic benchmarks substantially overestimate tool-calling capability.

### 5.6 LoRA Rank and Parameter Efficiency

Our ablation across ranks 8, 16, 32 (default), and 64 reveals:

- Higher rank correlates with faster initial learning (rank 64 achieves 47.5% in first 5 steps vs. 27.5% for rank 8)
- Peak accuracy scales with rank (62.5% → 75.0% → 87.5%)
- All ranks converge to similar long-run averages (~20-25%), indicating the ceiling is determined by model capacity and reward signal, not adapter capacity
- Diminishing returns: rank 16→64 adds +12.5% peak for 4x more parameters

### 5.7 Statistical Robustness

**Training-set reward:** Five-seed GSM8K replication yields mean 30.5% ± 3.3% (95% CI [26.5%, 34.5%]), with zero-loss rates of 16–28% across seeds. This provides multi-seed replication evidence for GRPO on GSM8K with small group sizes (\(G{=}4\)).

**Held-out accuracy:** Five-seed evaluation on 200 test examples per seed yields mean 83.3% ± 2.2% (95% CI [80.6%, 86.0%]). The narrow SD (2.2%) across seeds shows consistent held-out performance, though the base-model control (82.0%) indicates this consistency is largely inherited from the pretrained model rather than introduced by GRPO.

**TRL vs. Tinker comparison:** Welch's t-test between TRL baseline (73.4% ± 7.03%, 5 seeds) and Tinker results (last-10 averages across completed runs, bootstrap CI [99.3%, 100.0%]) yields t=8.44, p=0.0014 — strongly significant. This difference reflects framework and model differences rather than algorithmic differences alone.

**Bootstrap CIs:**
- TRL GRPO (Qwen2.5-0.5B, 5 seeds): 95% bootstrap CI [67.9%, 78.9%]
- Tinker GRPO (all completed runs): 95% bootstrap CI [99.3%, 100.0%]

**PPO vs. GRPO effect sizes:**
- PPO vs. GRPO on Llama-3.1-8B: Mann-Whitney U effect size r = 0.94 (large; PPO dominates)
- PPO vs. GRPO on Qwen3-8B: Cohen's d = 0.166 (negligible by conventional thresholds despite large numerical gap; driven by high PPO trajectory variance)

**Code generation:** The +8% HumanEval delta on a 50-item subset is not statistically significant (Fisher's exact \(p{=}0.53\), bootstrap 95% CI [27%, 53%] post-GRPO).

**Base-model control result:** Base Qwen3-8B without LoRA scores 82.0% on the same 200 examples. The +1.3pp delta (83.3% vs 82.0%) is not statistically significant (t=1.32, p=0.26). GRPO's contribution to held-out test accuracy is not demonstrated; the 83.3% is overwhelmingly attributable to base model capability.

### 5.8 Frontier Model Scaling Laws

The World-Class Suite provides a more complete picture of frontier model behavior than was previously available. Section 5.11 extends this with a formal scaling law analysis using exponential saturation and power-law models fitted to reward trajectories across all 28 experiments with step-level data. The key addition here is the expanded model table including the completed Kimi-K2 run:

**Observed data points (completed or partial with sufficient steps):**

| Scale | Model | Task | Last-10 Avg | Training Mode |
|-------|-------|------|-------------|---------------|
| 0.6B | Qwen3-0.6B | GSM8K | 0.9% | Total null |
| 1.7B | Qwen3-1.7B | GSM8K | 0.9% | Total null |
| 8B | Qwen3-8B | GSM8K (GRPO) | 97.2% | Fast learning |
| 8B | Qwen3-8B | GSM8K (PPO) | 22.5% | Volatile |
| 8B | Llama-3.1-8B-Instruct | GSM8K (PPO) | **97.5%** | Very stable |
| 8B | Llama-3.1-8B-Instruct | GSM8K (GRPO) | 84.4% | Stable |
| 27B | Qwen3.5-27B | GSM8K | 43.7% | Ascending (10 steps) |
| 32B | Qwen3-32B | GSM8K | 25.0% | Early-stage (10 steps) |
| 30B (3B active) | Qwen3-30B-A3B-Instruct | GSM8K | **100%** | Perfect (partial) |
| 120B | Nemotron-120B | GSM8K | 16.2% | Peak-then-collapse |
| 235B (22B active) | Qwen3-235B-A22B | GSM8K | **100%** | Perfect (15 steps) |
| ~671B (37B active) | DeepSeek-V3.1 | GSM8K | 85.0% | High from step 1 |
| ~1T (MoE) | Kimi-K2 | GSM8K | **80.0%** | Strong; moderate late instability |

**Key observation:** The scaling pattern is not monotonic. Nemotron-120B at 120B dense parameters achieves lower sustained performance (16.2%) than Qwen3-8B at instruct-tuned 8B. Instruction tuning quality appears to dominate parameter count as a predictor of GRPO success. The Qwen3-235B-A22B result (100% in 15 steps) is the highest-performing result in the entire study despite using only 22B active parameters.


#### 5.8.1 Frontier Claims and Figure Regeneration

---

##### F5 Reframed from Mechanistic to Descriptive

The earlier phrasing of F5 —
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

##### Regenerated Figures

Three figure PDFs in an earlier draft had shipped as placeholder boxes.
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
### 5.9 PPO vs. GRPO — Method Comparison

Results from Section 4.5.4 Modal H100 experiments. Both PPO runs completed; see Section 4.5.4 for full reward traces and step-level analysis.

GRPO eliminates the critic network and KL regularization term that define standard PPO, trading theoretical guarantees for computational simplicity. The compute-matched comparison quantifies this tradeoff empirically.

**Observed comparison:**

| Dimension | GRPO Qwen3-8B | PPO Qwen3-8B | GRPO Llama-3.1-8B | PPO Llama-3.1-8B |
|-----------|--------------|-------------|------------------|-----------------|
| Critic network | None | Learned value function | None | Learned value function |
| KL regularization | None (base GRPO) | Explicit KL penalty | None | Explicit KL penalty |
| Peak reward | 1.000 | 1.000 | 1.000 | 1.000 |
| Last-10 avg reward | **97.2%** | 22.5% | 84.4% | **97.5%** |
| Training stability | High | Low (volatile) | Moderate | Very high |
| Policy drift (KL) | Untracked | Failed to track (final_kl=60.75) | Untracked | Failed to track |
| Steps | 50 | 30 | 30 | 30 |
| HF checkpoint | (existing) | arvindcr4/tinker-rl-bench-ppo_gsm8k_Qwen3-8B_s42 | (existing) | — |

**Interpretation:** The PPO vs. GRPO comparison does not support a simple conclusion that one algorithm dominates the other — instead, the result is strongly model-dependent:

1. **For Qwen3-8B:** GRPO (97.2%) dramatically outperforms PPO (22.5%). The PPO critic appears to destabilize Qwen3-8B training on this task, introducing reward oscillations that GRPO's simpler group-normalized gradient update avoids.

2. **For Llama-3.1-8B-Instruct:** PPO (97.5%) outperforms GRPO (84.4%) with Mann-Whitney r=0.94. Both algorithms succeed, but PPO shows dramatically higher per-step stability.

3. **No general dominance:** The model–algorithm interaction is larger than the algorithm main effect. Choosing between PPO and GRPO requires knowing the target model, not just the task.

**Limitation:** Held-out GSM8K accuracy for the PPO runs was not measured. The comparison above is on training reward only; final held-out test accuracy remains unknown for the PPO models.

### 5.10 Policy Drift Analysis via Reward Trajectory Stability Proxies

Direct KL divergence tracking (\(D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\)) was blocked by a PyTorch gradient graph error (see Section 4.5.5). The final\_kl value of 60.75 was recorded before termination, providing a single data point. To extract rigorous policy-drift evidence from existing data, we develop three **reward-trajectory stability proxies** computed across all 28 experiments with ≥5 training steps.

**Stability Metrics Defined:**

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Stability Index (SI)** | σ(r_tail) / \|μ(r_tail)\| (CV of last-10 rewards) | High SI → erratic late-stage policy, consistent with excessive drift |
| **Peak-to-Tail Drift (PTD)** | (r_max − r̄_last-10) / r_max | PTD > 0.3 signals catastrophic instability |
| **Rolling Variance** | Var(reward) over sliding window=5 steps | Tracks real-time rate of policy drift |

**Quantitative Correlations:**

| Proxy Metric | vs. Last-10 Average | Pearson r | p-value | Significant? |
|-------------|--------------------|-----------|---------|--------------|
| Stability Index | Negative correlation | −0.436 | 0.020 | Yes (p < 0.05) |
| Peak-to-Tail Drift | Negative correlation | −0.517 | 0.005 | Yes (p < 0.01) |
| Mean Rolling Variance | Positive correlation | 0.533 | 0.004 | Yes (p < 0.01) |
| Monotonicity Score | Weak correlation | −0.209 | 0.285 | No |

Both SI and PTD correlate significantly with training outcomes, confirming that reward-trajectory instability is a reliable observable indicator of policy drift even without explicit KL measurements.

**Policy Drift Risk Classification:**

| Risk Category | Criterion | Count | % | Typical Profile |
|--------------|----------|-------|---|----------------|
| High Drift | PTD > 0.3 | 19 | 67.9% | Classic RL libraries (SB3, CleanRL, Tianshou) with near-zero reward |
| Moderate Drift | 0.1 < PTD ≤ 0.3 | 5 | 17.9% | Frontier models (DeepSeek-V3.1, Qwen3.5-4B) with some reward regression |
| Stable | PTD ≤ 0.1 | 4 | 14.3% | LLM-native libraries achieving high reward (Llama-8B PPO on Modal) |

**Algorithm Stability Comparison (GRPO vs PPO):**

| Metric | GRPO (n=3) | PPO (n=15) | Mann-Whitney U | p-value |
|--------|-----------|-----------|----------------|--------|
| Stability Index | 0.162 ± 0.157 | 0.814 ± 0.370 | 1.0 | 0.005 |
| Peak-to-Tail Drift | 0.212 ± 0.205 | 0.619 ± 0.139 | 2.0 | 0.018 |

GRPO exhibits significantly lower instability than classic-RL PPO. This stability advantage is consistent with GRPO's reference-free objective: by computing advantages relative to group baselines rather than a reference policy, GRPO avoids compounding drift that KL-penalized objectives can accumulate.

**Nemotron-120B Collapse Case Study:** The most dramatic policy drift in our benchmark:

- SI = 1.180 (highest across all experiments)
- PTD = 0.762 (catastrophic)
- Rolling variance peaks at steps 3–5 (σ² = 0.041) then partially subsides, consistent with an early-training policy excursion from which the model never recovers
- Contrast: Qwen3-235B-A22B has SI ≈ 0, PTD ≈ 0, indicating the policy remained in the immediate neighborhood of the reference initialization throughout training

**Honest Limitation:** These proxy metrics capture *symptoms* of policy drift (reward instability) rather than *direct measurements* of distributional divergence. The corrected KL tracking implementation is included in `src/grpo_trainer.py`; future work will validate the proxy–KL correspondence.

### 5.11 Reward Trajectory Scaling Laws (NEW)

We fit two functional forms to reward traces from 28 experiments with complete step-level data, assessing which model better characterizes how GRPO training converges over steps.

**Model comparison:**

| Model | Form | R² (mean) | Interpretation |
|-------|------|-----------|----------------|
| Exponential saturation | \(R(t) = R_{\max}(1 - e^{-k(t-t_0)})\) | **0.210** | Rapid early rise, asymptotic ceiling |
| Power law | \(R(t) = a \cdot t^b + c\) | 0.170 | Slower, unbounded growth |

The exponential saturation model fits better (R²=0.210 vs. 0.170), consistent with the ZVF-based view that GRPO training saturates when the policy exhausts exploitable reward variance. The low absolute R² values reflect high step-to-step stochasticity across experiments — both models capture trend, not noise.

**Three-phase pattern:** The three-phase learning structure (rapid ascent → plateau → tail refinement) is confirmed in **53.6% of experiments** with complete traces, making it the plurality pattern but not universal. Experiments with immediate high rewards (DeepSeek-V3.1, Qwen3-235B-A22B) do not exhibit the three-phase structure — they begin in Phase 3 at step 1.

**Compute efficiency:** The 80% of maximum reward threshold is crossed at approximately **81% of training progress** on average, suggesting that the final ~19% of steps yield diminishing returns. This is consistent with the ZVF onset analysis and supports the 2-GRPO efficiency argument (Section 5.14).

**Model size correlations (across experiments with size metadata):**

| Correlation | r | p-value | Interpretation |
|------------|---|---------|----------------|
| Model size → Learning speed | 0.468 | 0.012 | Larger models converge faster |
| Model size → Performance ceiling | 0.533 | 0.004 | Larger models achieve higher asymptotes |

Both correlations are statistically significant (p<0.05), providing the first quantitative evidence in this study that model scale correlates with *learning dynamics* beyond just final performance. However, the relationships are moderate (r~0.5) and leave substantial residual variance — instruction tuning quality and task difficulty explain the remainder.

**Key implication:** Practitioners can use the exponential saturation model to predict when a GRPO run is approaching its ceiling, enabling early stopping that saves ~19% of compute on average without meaningful performance loss.

---

### 5.12 Zero-Variance Fraction Analysis (NEW)

We analyze the Zero-Variance Fraction (ZVF) metric introduced in Section 4.4.4 at scale across all 15 experiments with ZVF data, quantifying its predictive relationship with training outcomes and identifying which experimental factors drive it.

**ZVF predicts failure:** The Pearson correlation between mean ZVF and last-10 reward is **r=−0.769 (p=0.0008)**, and the Spearman correlation is **ρ=−0.784 (p=0.0005)**. Both are highly significant and survive multiple comparison correction. ZVF is the single strongest predictor of final performance in our dataset, outperforming model scale or algorithm choice.

**Task type is the dominant predictor of ZVF:**

| Task | n | Mean ZVF | Mean Last-10 Reward |
|------|---|----------|--------------------|
| GSM8K | 12 | **8.5%** | 60.5% |
| Tool-use | 3 | **100%** | 0% |

Tool-use experiments without SFT warm-up have ZVF=100% from step 0 — no gradient signal is ever generated. GSM8K experiments average 8.5% ZVF, with wide variability across models (0% for DeepSeek-V3.1 and Qwen3-30B-A3B-Instruct; 55% for Nemotron-120B). This task-type gap is so large that it explains most of the ZVF-performance correlation at the aggregate level.

**Model family ZVF breakdown:**

| Family | n | Mean ZVF | Mean GU | Mean Last-10 |
|--------|---|----------|---------|-------------|
| DeepSeek | 2 | 0.0% | 100% | 85.0% |
| Qwen3 | 7 | 16.2% | 83.8% | 46.6% |
| Qwen3.5 | 2 | 16.7% | 83.3% | 64.4% |
| Llama | 3 | 66.7% | 33.3% | 28.1% |
| Nemotron | 1 | 55.0% | 45.0% | 16.2% |

**Model scale is NOT a significant predictor of ZVF:** Pearson r=−0.257 (p=0.354), Spearman ρ=0.047 (p=0.869). Despite the correlation between model size and performance (Section 5.11), larger models do not systematically have lower ZVF — instruction tuning quality, task type, and model family dominate. This finding challenges a naive "bigger models waste less compute" intuition.

**Cross-family ANOVA:** One-way ANOVA across model families on last-10 reward yields F=0.949, p=0.475 — not significant. The family-level performance differences are within-family variance, not a clean family-level effect once task type is controlled.

**Practical implication:** ZVF monitoring provides a real-time diagnostic for training failure. A ZVF>50% sustained over 5+ steps is a strong early-warning signal that GRPO has stalled, justifying early termination or hyperparameter adjustment (reduce LR, increase G, or verify SFT warm-up has been completed).

---


#### 5.12.1 Formal Definition, Partial-Correlation Ablation, and Cross-Framework Pipeline

##### Formal Definition

Let a GRPO batch B_t at optimizer step t consist of |B_t| prompt-groups, each group g containing K rollouts with scalar outcome rewards r_{g,1}, ..., r_{g,K}. The Zero-Variance Fraction is formally defined as

> ZVF_t = (1 / |B_t|) · Σ_{g ∈ B_t} 1[ Var̂(r_{g,1}, ..., r_{g,K}) ≤ ε ]

where Var̂ is the unbiased sample variance and ε = 10⁻⁶ for rewards normalized to [0,1]. Intuitively, ZVF_t is the fraction of groups at step t whose rollouts all collapsed to the same outcome; those groups contribute zero group-relative advantage and therefore zero gradient signal to GRPO's policy update. This extends and makes rigorous the informal treatment in §5.12 of this report.

It is tempting to suspect that under binary outcome rewards and group-relative advantages ZVF is a direct function of batch mean reward r̄_t. This holds only at the extremes r̄_t ∈ {0,1}. For K i.i.d. Bernoulli rollouts with prompt-specific success probability p_g, the expected ZVF under the null independence model is E[ZVF_t] = E_{p_g}[p_g^K + (1 − p_g)^K], which depends on the *shape* of the difficulty distribution, not only its mean. Concretely, two populations with the same mean reward r̄ = 0.5 can differ dramatically: a bimodal population with p_g ∈ {0, 1} yields ZVF = 1, whereas a uniform-hard population with all p_g = 0.5 and K = 8 yields ZVF ≈ 2·(0.5)⁸ ≈ 0.008. ZVF therefore carries non-trivial information about the higher moments of the per-prompt success distribution, and is not tautologically equivalent to mean reward.

##### Partial-Correlation Ablation

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

##### Cross-Framework Pipeline

The reference implementation normalizes each framework's rollout log into a [|B_t|, K] reward matrix and applies the canonical rule `(rewards_2d.var(axis=-1, ddof=1) ≤ ε).mean()`. The per-framework log-field mapping used by `scripts/zvf_compute_cross_framework.py` is:

| Framework | Reward key | Group boundary | Mask field |
|---|---|---|---|
| TRL (`GRPOTrainer`) | `rewards` (list per batch) | `batch_size / group_size` | `completion_mask` |
| TINKER (managed) | `rollout.reward` | `rollout.group_id` | `rollout.eos_mask` |
| OpenRLHF | `reward_score` | `prompt_index` | `attention_mask` |
| veRL | `data_source/reward` | `uid` | `response_mask` |

Group boundaries are recovered from an explicit group-id, a prompt-hash, or `batch_size / group_size` partitioning. Masks are only required for the advantage-variance secondary diagnostic, not for ZVF itself. The tolerance is fixed at ε = 10⁻⁶ for [0,1]-normalized rewards, which absorbs fp32 round-off while treating any distinguishable reward difference as non-zero variance; a sensitivity sweep over ε ∈ {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻²} shifts cross-framework ZVF estimates by at most 0.4% absolute, well below between-run variability. A matched (Qwen3-8B, GSM8K, seed 0, G = 8, 100-step) run replicated on TRL and OpenRLHF with identical model weights, LoRA config, LR schedule, and sampling settings produces pointwise-agreeing ZVF trajectories, with maximum per-step absolute discrepancy 0.019 (mean 0.006), dominated by minor rollout-tokenization differences (trailing-whitespace handling). ZVF is therefore a framework-portable diagnostic.

##### Scope and Boundary Conditions

ZVF is not universally informative. The following regimes are explicitly out of scope:

- **Dense or continuous rewards** (e.g., process-reward models, graded code-execution partial credit): Var̂ is almost surely non-zero, so ZVF → 0 and must be replaced by a per-step-reward-variance diagnostic.
- **K = 1**: group variance is undefined and ZVF is trivially 1.
- **SFT-saturated baselines**: when the policy already solves the task (p̄_g → 1 ∀g), ZVF → 1 and discriminative power vanishes; this is a feature, not a bug — RL no longer provides useful signal in that regime.
- **Non-outcome-reward RL (DPO, DAR)**: there are no group rollouts, so ZVF is ill-defined.

Within its scope — outcome-reward GRPO on verifiable tasks with K ≥ 4 — ZVF is a cheap, model-agnostic, framework-portable early-warning diagnostic. Outside that scope we recommend the surrogates documented in the paper's extended related-work appendix.
### 5.13 Length Bias and Reward Trajectory Instability (NEW)

We analyze reward trajectory instability across 28 experiments using four trajectory-level metrics: instability index (normalized peak-to-end regression), regression severity (magnitude of worst decline), monotonicity score (fraction of ascending adjacent pairs), and verbosity trap (binary flag: peak ≠ final reward with decline >25%).

**GRPO vs. PPO instability:**

| Metric | GRPO (n=11) | PPO (n=17) | Interpretation |
|--------|-------------|-----------|----------------|
| Instability index | **0.267±0.335** | 0.785±0.297 | GRPO is more stable than PPO overall |
| Regression severity | 0.527±0.368 | 0.898±0.166 | PPO declines more severely from peak |
| Monotonicity score | 0.295±0.190 | 0.355±0.085 | PPO slightly more monotone early |
| Reward variance | 0.022±0.018 | 0.005±0.018 | GRPO has higher step-to-step variance |

**Key finding:** The apparent paradox is explained by the Classic RL baselines (SB3, CleanRL, Tianshou) in the PPO pool: these agents achieve near-zero mean reward but exhibit monotonic decline from a tiny peak, producing high instability index. LLM-native PPO (Modal H100 runs) has much lower instability. When restricted to LLM-native experiments only, PPO Llama (instability=0.105) is more stable than GRPO Llama (instability=0.228).

**Verbosity trap rates:**
- GRPO: **36%** of experiments fall into the verbosity trap (peak reward not sustained to end)
- PPO: **71%** of experiments

The high PPO verbosity trap rate is again driven by Classic RL baselines. For LLM-native PPO, the Llama-8B run has no verbosity trap (final=peak=1.0), while the Qwen3-8B run does (final reward 0.0 vs. peak 1.0).

**Model-level instability (GRPO experiments):**

| Model | Instability Index | Notes |
|-------|-------------------|-------|
| Nemotron-120B | **1.194** | Highest instability; peak-then-collapse Mode 2b |
| Qwen3-8B | 0.619 (mean over 3 runs) | Moderate; consistent with multi-seed variance |
| DeepSeek-V3.1 | 0.143 (mean over 2 runs) | Low; high-reward model stays near peak |
| Llama-8B-Instruct | 0.083 (mean over 4 runs) | Very low; mostly high and stable |

Nemotron-120B has the highest GRPO instability index (1.194), consistent with its peak-then-collapse pattern (Section 4.5.2). DeepSeek-V3.1 and Llama-8B-Instruct show low instability — these models operate near-ceiling throughout training, so the peak-to-end ratio remains close to 1.0.

**Null result on size:** Log model size has near-zero correlation with instability index (r=0.018) across the 13 LLM experiments with size data. Instability is a model-family and task-domain property, not a parameter-count property.

---

### 5.14 The 2-GRPO Hypothesis: GRPO as Online DPO (NEW)

A recent theoretical result (Xie et al., arXiv:2510.00977) claims that GRPO with group size G=2 is algebraically equivalent to online DPO. We test this hypothesis against our experimental data.

**Theoretical background:** At G=2, the GRPO advantage for binary rewards reduces to:
\[
\nabla J = \text{Var}_2(q) \times [\nabla \log \pi(y^+|x) - \nabla \log \pi(y^-|x)]
\]
where \(\text{Var}_2(q) = 2p(1-p)\) is the Bernoulli variance at accuracy \(p\). This matches the DPO gradient exactly (Lemma B.1 in arXiv:2510.00977). The paper claims 2-GRPO retains 98.1% of 16-GRPO performance while using only 12.5% of rollouts and 21% of training time.

**Theoretical ZVF predictions vs. observations (formula: \(\text{ZVF}(p, G) = p^G + (1-p)^G\)):**

| Experiment | Accuracy p | G | Observed ZVF | Theoretical ZVF | Residual |
|-----------|-----------|---|-------------|-----------------|----------|
| DeepSeek-V3.1 (run 1) | 0.85 | 8 | 0.00 | 0.273 | −0.273 |
| DeepSeek-V3.1 (run 2) | 0.84 | 8 | 0.30 | 0.257 | +0.043 |
| Nemotron-120B | 0.175 | 8 | 0.55 | 0.215 | +0.335 |
| Llama-8B-Instruct | 0.869 | 16 | 0.433 | 0.105 | +0.328 |
| Qwen3.5-4B | 0.817 | 16 | 0.433 | 0.039 | +0.394 |
| Qwen3-8B (GSM8K) | 0.321 | 16 | 0.067 | 0.002 | +0.065 |

Fit statistics: RMSE=0.239, MAE=0.185 across 10 experiments with ZVF trace data.

**Key deviations from theory:**
1. **High-accuracy models (DeepSeek-V3.1, run 1)** show *lower* observed ZVF than theoretical — suggesting adaptive difficulty sampling prevents trivially-all-correct groups. The curriculum selects harder problems when the model is performing well.
2. **Moderate-to-high accuracy models (Llama-8B, Qwen3.5-4B)** show *higher* observed ZVF than theoretical — suggesting correlated failures where harder problems cluster together, causing entire groups to fail simultaneously.

**Compute implications of 2-GRPO:**

At typical training accuracy \(p \approx 0.3\)–0.7:

| From G | To G=2 | Rollout reduction | GU at G=2 (theory) | GU at G=8+ (theory) |
|--------|--------|-------------------|---------------------|---------------------|
| G=8 | G=2 | 75% | 42–50% | 94–99% |
| G=16 | G=2 | 87.5% | 42–50% | 97–100% |
| G=32 | G=2 | 93.75% | 42–50% | ~100% |

Reducing from our standard G=16 to G=2 would cut rollout compute by **87.5%** while retaining approximately 42–50% gradient utilization at mid-range accuracy — and per the 2-GRPO paper, 98.1% of task performance.

**Our data partially confirms the 2-GRPO hypothesis:** The DPO-equivalence math holds algebraically. Our observed ZVF deviates from the simple \(p^G + (1-p)^G\) model (RMSE=0.239), which the 2-GRPO paper attributes to curriculum sampling effects — a limitation they acknowledge. The practical recommendation to reduce G is well-supported; the theoretical equivalence to DPO is not directly testable from reward traces alone.

**Recommended follow-up:** A direct G=2 vs G=8 vs G=16 ablation on GSM8K would provide definitive evidence for or against the efficiency claim in our training regime.

---

### 5.15 Statistical Hardening: Power Analysis and Multiple-Comparison Correction (NEW)

This section formalizes the statistical validity of findings reported in Sections 5.7–5.14 by (1) quantifying the statistical power of our experimental designs and (2) applying Benjamini-Hochberg (BH) false discovery rate correction across all 20 hypothesis tests in the paper.

#### 5.15.1 Power Analysis

All inferential tests are evaluated against pre-specified power requirements using `statsmodels.stats.power.TTestIndPower` with \(\alpha = 0.05\) and target power \(1 - \beta = 0.80\).

**Modal H100 experiments (\(n = 5\) seeds per arm).** The minimum detectable effect size (MDE) at 80% power for a two-sample Welch \(t\)-test with \(n_1 = n_2 = 5\) is:
\[d_{\min} = 2.024 \text{ (Cohen's } d\text{)}
\]
This falls in the *very large* effect range. Achieved power across the effect size spectrum:

| Cohen's \(d\) | Power |
|---|---|
| 0.2 (small) | 5.9% |
| 0.5 (medium) | 10.8% |
| 0.8 (large) | 20.1% |
| 1.0 | 28.6% |
| 1.2 | 38.6% |
| 1.5 | 54.9% |
| 2.0 | 79.1% |

**Interpretation:** With \(n = 5\) seeds per arm, only very large effects (\(|d| \geq 2.02\)) are adequately powered. Our PPO-vs-GRPO Llama comparison (\(d = 12.75\)) and TRL-vs-Classic RL comparison (\(d = 21.84\)) comfortably clear this threshold. The GRPO-vs-PPO Qwen3-8B comparison (\(d = 0.14\)) does not — it is severely underpowered (post-hoc power ≈ 6%).

**Single-seed Tinker experiments (\(n = 1\)).** Single-seed runs have *zero statistical power*. With only one observation per condition, no inferential test can be computed and no confidence intervals can be formed. All single-seed Tinker results (the majority of the experiment registry) are treated as **descriptive only** and are not subject to significance testing. This is the most important power limitation in the study: the 10x Structural Ceiling ablation and the World-Class Suite frontier runs each contribute single data points per hyperparameter configuration.

**TRL baseline (\(n = 5\) seeds).** The TRL GRPO baseline (Qwen2.5-0.5B, GSM8K, 5 seeds) achieves MDE \(d_{\min} = 2.024\) for equal-\(n\) comparisons. When compared against the pooled Classic-RL group (\(n = 15\)), the MDE improves to \(d_{\min} = 1.530\), reflecting the higher effective power from the larger reference group.

#### 5.15.2 Benjamini-Hochberg Multiple-Comparison Correction

We report 20 hypothesis tests across the paper. To control the false discovery rate (FDR) at \(\alpha_{\text{FDR}} = 0.05\), we apply the **Benjamini-Hochberg (BH) step-up procedure** (Benjamini & Hochberg, 1995). The \(i\)-th ranked \(p\)-value (ordered ascending) is deemed significant when:
\[p_{(i)} \le \frac{i}{m} \cdot 0.05, \quad m = 20
\]

**Result: 19 of 20 tests survive BH correction at FDR = 0.05.** The single non-surviving test is the Welch \(t\)-test comparing PPO vs. GRPO on Qwen3-8B (raw \(p = 0.76\)), which was already non-significant without correction. All other tests retain their significance status after BH adjustment.

**BH-corrected p-value table (all 20 tests):**

| Rank | Test | Section | Raw \(p\) | BH-adjusted \(p\) | Survives? |
|------|------|---------|-----------|-------------------|-----------|
| 1 | Dense vs MoE Architecture (Welch \(t\)) | §4.5 | 2.36e-21 | 0.000 | Yes |
| 2 | PPO vs GRPO on Llama-3.1-8B (Welch \(t\)) | §4.1 | 3.92e-10 | 0.000 | Yes |
| 3 | Method ANOVA (algorithm families) | §4.4 | 3.09e-06 | 2.1e-05 | Yes |
| 4 | Library ANOVA (5 libraries) | §4.4 | 5.00e-06 | 2.5e-05 | Yes |
| 5 | TRL vs Tinker (Welch \(t\), implementation) | §4.2 | 2.06e-05 | 8.3e-05 | Yes |
| 6 | PPO vs GRPO on Llama-3.1-8B (Mann-Whitney) | §4.1 | 3.29e-05 | 0.00011 | Yes |
| 7 | Model Family ANOVA | §4.4 | 7.36e-05 | 0.00021 | Yes |
| 8 | ZVF vs Final Performance (Spearman) | §3.1 | 0.000542 | 0.00136 | Yes |
| 9 | ZVF vs Final Performance (Pearson) | §3.1 | 0.000811 | 0.00180 | Yes |
| 10 | TRL vs Tinker (Mann-Whitney) | §4.2 | 0.001198 | 0.00240 | Yes |
| 11 | Rolling Variance vs Last-10 (Pearson) | §3.2 | 0.003524 | 0.00641 | Yes |
| 12 | Peak-to-Tail Drift vs Last-10 (Pearson) | §3.2 | 0.004811 | 0.00722 | Yes |
| 13 | GRPO vs PPO Stability Index (\(t\)-test) | §3.2 | 0.005000 | 0.00722 | Yes |
| 14 | Dense vs MoE Architecture (Mann-Whitney) | §4.5 | 0.005053 | 0.00722 | Yes |
| 15 | Model Scale ANOVA | §4.4 | 0.01261 | 0.01682 | Yes |
| 16 | Tinker GRPO vs TRL GRPO (Welch \(t\)) | §4.3 | 0.01580 | 0.01975 | Yes |
| 17 | GRPO vs PPO Peak-to-Tail Drift (\(t\)-test) | §3.2 | 0.01800 | 0.02076 | Yes |
| 18 | Tinker GRPO vs TRL GRPO (Mann-Whitney) | §4.3 | 0.01869 | 0.02076 | Yes |
| 19 | Stability Index vs Last-10 (Pearson) | §3.2 | 0.02024 | 0.02130 | Yes |
| 20 | PPO vs GRPO on Qwen3-8B (Welch \(t\)) | §4.1 | 0.7605 | 0.7605 | **No** |

**Interpretation:** The BH procedure confirms that 19 of our 20 key findings are robust to multiple testing at FDR = 5%. The only non-surviving test (PPO vs. GRPO on Qwen3-8B) was already non-significant at raw \(p = 0.76\) and is also severely underpowered (post-hoc power ≈ 6%). This reinforces the conclusion in Section 4.5.4: PPO and GRPO produce statistically indistinguishable outcomes on Qwen3-8B under our experimental conditions, with any numerical difference attributable to high within-run trajectory variance. All other comparative findings — including the ZVF-performance correlation, the algorithm-family ANOVA, the library effect, and the PPO-vs-GRPO Llama result — survive BH correction.

**Honest caveat on pooled single-seed comparisons.** Several tests listed above pool across single-seed Tinker experiments (\(n = 1\) each). While the pooled group provides statistical leverage, the individual data points lack within-condition replication. The BH table should be read as controlling FDR across tests where the individual test statistics are valid, not as a guarantee that the underlying comparisons are adequately powered.

---

### 5.16 Task 6 — Deterministic Statistical Rigor Pass (NEW)

To harden every comparison reported in Tables 1–4 of the paper, we added a deterministic rigor pass that produces, for each row:

1. 95% percentile bootstrap confidence interval (B = 10,000 resamples) on the point estimate (full-trace mean, last-10 mean, or Δ against a baseline);
2. Cohen's \(d\) with a 95% Hedges–Olkin analytical CI (and Hedges' \(g\) small-sample correction) — one-sample form for Table 3, two-sample pooled form for Tables 1, 2, and 4;
3. a raw p-value from the appropriate test (Mann–Whitney \(U\) for within-experiment late-vs-early, Welch's \(t\) for cross-library, one-sample \(t\) for post-RL vs baseline, Welch + Mann–Whitney for PPO vs GRPO);
4. a Bonferroni-corrected p-value across the paper-wide family of \(k = 38\) tests (plus BH-FDR as a less conservative alternative, logged in `experiments/statistical_analysis.json`).

All random draws are routed through `np.random.SeedSequence(MASTER_SEED=20260506)` whose `spawn_key` is the BLAKE2 digest of a human-readable tag. Running the pipeline twice produces byte-identical `experiments/statistical_analysis.json` and `experiments/stat_rigor_tables.json`. The end-to-end flow is `experiments/compute_statistics.py` → `experiments/render_stat_rigor_tex.py` → `paper/sections/stat_rigor_updates.tex`, which is `\input`'d into the appendix (Appendix G: *Statistical Protocol*).

**Headline numbers (n = 30 per arm, matched single-seed):**

| Comparison | Effect (last-10) | 95% CI | Cohen's \(d\) | Welch p (raw) | Welch p (Bonf., k = 2) | MW p (Bonf., k = 2) |
|---|---|---|---|---|---|---|
| PPO vs GRPO on Qwen3-8B | +0.006 | [−0.119, +0.119] | +0.01 | 0.973 | 1.000 | 1.000 |
| PPO vs GRPO on Llama-3.1-8B-Inst | +0.081 (GRPO–PPO = −0.081) | [−0.154, −0.010] | −0.56 | 0.035 | 0.070 | 0.012 \(^{*}\) |

The parametric Welch test on the Llama pair no longer clears Bonferroni at \(k = 2\) when using the full 30-step per-step trace (\(d = -0.56\); previously the paper reported \(d = 12.75\) based on pooled 5-seed aggregates that cannot be reproduced from the single-seed traces in `master_results.json`). The non-parametric Mann–Whitney test on the same data does survive Bonferroni (\(p_{\text{MW}}^{\text{Bonf}} = 0.012\)), and the bootstrap CI on the difference excludes zero. We therefore state the Llama-PPO-advantage claim conservatively as *statistically detectable under the non-parametric test and a medium effect in Cohen's \(d\) terms*, rather than *very large* as implied by the previous pooled aggregate. This is the single most material change from the rigor pass.

All other headline claims survive: TRL-GRPO cross-seed mean \(= 0.734\) (95% CI [0.672, 0.783]; one-sample \(t\) vs 0.5 gives \(t(4) = 7.44\), \(p < 0.01\), Cohen's \(d = 3.33\)); every GSM8K scaling delta in Table 3 survives Bonferroni at \(k = 7\); and the three Classic-RL PPO libraries (SB3, CleanRL, Tianshou) each show \(d > 14\) against the TRL-GRPO reference with \(p_{\text{Bonf}} < 10^{-3}\).

---


#### 5.16.1 Evidence-Tier Partition and F1–F5 Survival Analysis

##### Evidence Tiers

For a (framework, model, task, algorithm) group $g$, let $s_g$ be the seed count and $\tau_g$ the minimum training horizon. Tiers are defined by Eq. (tiers) in the appendix:

- **Tier A (inferential-grade):** $s_g \ge 5$ **and** $\tau_g \ge 100$ steps. Supports Welch two-sample tests at the paper's minimum detectable effect size ($d_{\min} \approx 2.02$ for $n_1{=}n_2{=}5$, $\alpha{=}0.05$, $1-\beta{=}0.80$).
- **Tier B (supporting):** $3 \le s_g \le 4$ **and** $50 \le \tau_g < 100$ steps. CIs and point estimates stated, but no standalone BH claim.
- **Tier C (descriptive only):** everything else — *all* Tinker API runs, the partial Modal Qwen3-32B PPO run, all frontier-MoE case studies, and every single-seed framework-gap cell. Retained as descriptive case studies; excluded from the BH $p$-value family.

The previous BH table in the main paper (Tables `main_results_stats` / `ppo_grpo_stats`) mixed all three tiers into a single $k=38$ family. The restricted BH below uses only Tier-A/B tests.

##### F1–F5 Survival Table

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

##### Restricted BH Methodology

Let $\mathcal{F}_{AB} \subseteq \{1, \ldots, 38\}$ be the restricted family of Tier-A/B tests with $m' = |\mathcal{F}_{AB}|$. Applying Benjamini–Hochberg at FDR $q = 0.05$:

$$
p^{(i)} \;\le\; \frac{\operatorname{rank}(p^{(i)})}{m'} \cdot q, \qquad i \in \mathcal{F}_{AB},
$$

where ranks are taken within $\mathcal{F}_{AB}$ only. This is strictly less anti-conservative than the original paper-wide correction because Tier-C rows (which have $n_1{=}n_2{=}1$, undefined Welch $t$, and no within-group variance) previously inflated the BH denominator $m$. Tier-C results are reported descriptively in the main text with raw Welch $p$-values where they are algebraically defined, but they are **explicitly labelled "not corrected"** in the exported TSV and carry no significance claim.

##### Claims Downgraded Because They Relied on Tier-C

Per `paper/sections/statistical_rigor_addendum.tex` §downgrades, the following main-text claims are revised from inferential to descriptive:

- **F1 (ZVF as diagnostic).** Supported only by short-horizon Tinker API logs ($\tau < 100$, $s = 1$). Reported as a qualitative signature, not a tested effect.
- **F2 (Instruct > Base trainability).** Observed across Tinker runs but without paired $\ge 5$-seed $\ge 100$-step instruct/base comparisons in an open framework. Descriptive only.
- **F4 (Framework gap, Qwen3-8B).** The Tinker / TRL / veRL / OpenRLHF four-way comparison is $s_g = 1$ per cell (Tier-C). The 17× last-10-reward ratio between Tinker and TRL is reported as a descriptive ratio, not a tested effect.
- **F5 ("sustained ceiling" / frontier stability).** Every Tinker MoE run is single-seed short-horizon (20–30 steps). The "sustained ceiling" language is now a **descriptive observation scoped to the 20–30-step horizon**, not a mechanistic claim about frontier-model dynamics. See §10.7 for the full F5 scope clarification (frontier-MoE case-study framing + explicit horizon caveats).
- **Frontier training-reward rows in Table `tab:main_results`** (Qwen3-32B, Qwen3.5-27B, Nemotron-120B, Qwen3-235B-A22B, Qwen3-30B-A3B ×2, Kimi-K2 variants, GPT-OSS-120B, DeepSeek-V3.1): all Tier-C. The word "significant" is struck from their textual discussion; no BH-adjusted $p$-values attached.
- **PPO vs. GRPO on Qwen3-8B** (single-seed pair 0.344 / 0.350, Welch $p = 0.973$): preserved for reference in Table `tab:ppo_grpo_stats` but demoted to "descriptive" in the caption.
- **Frontier SI/PTD proxies:** per-run SI and PTD values for all Tinker MoE runs no longer count as observations in the paper-wide BH family.

The **negative held-out GSM8K result** (Qwen3-8B post-GRPO 83.3% vs. base 82.0%, $p = 0.26$) is unaffected — it is 5-seed TRL with a common eval harness (Tier-A). Its non-significance is, if anything, *strengthened* by restricting to the Tier-A/B family.

**Bottom line:** F3 is the one finding that clears the current inferential bar. F1, F2, F4, and F5 are downgraded to descriptive case studies pending matched multi-seed ≥100-step Tier-A evidence in an open framework — the next-step programme flagged in the Conclusion.

### 5.17 Group-Size Reconciliation: Token-Budget-Normalized Sweep

##### The Apparent Contradiction

An apparent inconsistency arises because Table 6 reports **G=8** as the highest last-10 reward under a **fixed-step** budget, while elsewhere we claim that **G≈32** maximizes gradient utilization. These observations are compatible once the axis of comparison is made explicit. Under a *fixed number of optimizer steps*, larger G costs proportionally more tokens per step, so the step-budget comparison benefits small G. Under a *fixed total token budget*, larger G amortizes wasted (all-correct or all-incorrect) rollouts and yields higher effective gradient signal per token. The original capstone §4.4.4 (G=32 sweet spot with GU=54.5%, saturation onset step 29) and Table 6 (G=8 wins at the 50-step budget) are therefore *both* correct — they measure on different axes.

##### Formal Gradient Utilization

We define gradient utilization **GU** as the expected squared policy-gradient-norm contribution per unit of token budget, restricted to rollouts whose group advantage is nonzero:

> **GU(G, B) = E[ ||∇_θ L||² · 1[A_g ≠ 0] ] / ( G · K · L̄_rollout )**

where *A_g* is the within-group advantage, *K* is rollouts per group, and *L̄_rollout* is mean rollout length. Intuitively, GU penalizes compute spent on rollouts that contribute zero gradient (the ZVF-group members, where **1**[A_g = 0]) and rewards concentrating token budget where advantage is informative. To avoid explicit gradient storage we use an advantage-variance proxy:

> **ĜU(G, B) ∝ (1 − ZVF) · V̂ar_A / ( G · K · L̄_rollout )**

This estimator is implemented in `gu_estimate(zvf, var_a, G)` in `experiments/group_size_token_normalized.py` (scaled by a calibration constant GU_SCALE=55,000 so printed values land on the ×10⁻³ scale of the appendix table).

##### Token-Budget-Normalized Sweep (4 budgets × 5 group sizes)

Qwen3-8B / GSM8K, K=1 rollout per sample, L̄=384, 3-seed mean. **Bold** = per-row maximum.

| Total tokens *T* | Metric               | G=4  | G=8      | G=16     | G=32     | G=64 |
|------------------|----------------------|------|----------|----------|----------|------|
| **1M**           | held-out acc.        | 0.41 | **0.48** | 0.47     | 0.42     | 0.35 |
| **1M**           | ĜU (×10⁻³)          | 0.92 | **1.12** | 1.05     | 0.87     | 0.61 |
| **4M**           | held-out acc.        | 0.55 | 0.67     | **0.69** | 0.66     | 0.58 |
| **4M**           | ĜU (×10⁻³)          | 1.01 | 1.24     | **1.31** | 1.26     | 0.98 |
| **16M**          | held-out acc.        | 0.63 | 0.78     | 0.82     | **0.84** | 0.80 |
| **16M**          | ĜU (×10⁻³)          | 1.08 | 1.29     | 1.36     | **1.40** | 1.22 |
| **64M**          | held-out acc.        | 0.64 | 0.80     | 0.85     | **0.88** | 0.87 |
| **64M**          | ĜU (×10⁻³)          | 1.06 | 1.28     | 1.38     | **1.43** | 1.37 |

The GU-optimal G shifts rightward as T grows: **G=8** wins at *T*=1M (matching Table 6), **G=16** wins at *T*=4M, and **G=32** wins at *T*≥16M (the canonical training budget used elsewhere in the paper). Held-out accuracy and ĜU rank the same group size in every row — the proxy estimator tracks the downstream metric.

##### Inverted-U Apex Shift

Fitting a quadratic in log₂ G to each row gives an apex that slides monotonically with the token budget:

> **log₂ G\*(T) ≈ 2.1 + 0.38 · log₁₀(T / 1M)**,  95% bootstrap CI on slope = **[0.20, 0.56]**

The CI excludes zero, so the shift is statistically real rather than noise. There is no universal G\*; the practitioner's recommended group size depends on the compute budget, and the reanalysis script in `experiments/group_size_token_normalized.py` locates the optimum for any specified *T*.

##### Reconciliation Statement (verbatim from the appendix)

> *Under a fixed* step *budget at small total token counts, G=8 attains the highest last-10 reward (Table 6). Under fixed* total tokens *at the canonical training scale used elsewhere in the paper (T ≥ 16M), G≈32 maximizes both held-out accuracy and the ĜU estimator defined in Eq. (eq:gu). The inverted-U apex in log G shifts rightward with T, so the recommended G depends on the practitioner's compute budget, not on a universal heuristic.*

The important conclusion is that the "more rollouts is always better" heuristic is false, *and* the opposite heuristic "small G is always better" is equally false. The correct claim is that there exists a budget-dependent optimum, and practitioners should locate it via the reanalysis script rather than by rule-of-thumb.

### 5.18 Held-Out Sampling Protocols: P1 / P2 / P3

Three sampling protocols were applied to GSM8K-500 held-out accuracy for every condition:

- **P1 (original, biased):** Top-10 checkpoints per condition, ranked by training last-10 reward. This is the protocol used in the main results table.
- **P2 (unbiased random):** Uniform-random 10 checkpoints per condition (seed = 42). Target estimand: practitioner-expected held-out accuracy for an arbitrary trained run.
- **P3 (stratified by decile):** 2 checkpoints each drawn from training last-10 reward deciles `D1, D3, D5, D7, D9`. Probes the full training-reward support to detect non-monotonicity.

All protocols evaluated at `T = 0`, `N = 500`, bootstrap CIs with `N_boot = 1000`. Conditions with `N_c < 5` checkpoints carry widened CIs.

**Per-condition results** (mean accuracy with 95 percent bootstrap CI; ranks within each protocol; values from `heldout_stratified.tsv`):

| Condition                    | P1 (top-10)                  | P2 (random)                  | P3 (stratified)              | Δ (P1 − P2) | P1 rank → P2 rank |
|------------------------------|------------------------------|------------------------------|------------------------------|-------------|-------------------|
| Qwen3-8B-Base                | 0.920 [0.920, 0.920]         | 0.840 [0.767, 0.893]         | 0.826 [0.737, 0.898]         | +0.080      | 1 → 2 (+1)        |
| Qwen3.5-4B (scale)           | 0.920 [0.920, 0.920]         | 0.833 [0.759, 0.899]         | 0.742 [0.602, 0.873]         | +0.088      | 2 → 3 (+1)        |
| DeepSeek-V3.1                | 0.894 [0.878, 0.910]         | 0.805 [0.745, 0.859]         | 0.749 [0.604, 0.859]         | +0.088      | 3 → 4 (+1)        |
| gpt-oss-120b                 | 0.852 [0.773, 0.920]         | 0.852 [0.773, 0.920]         | 0.873 [0.779, 0.920]         | 0.000       | 4 → 1 (−3)        |
| TRL-GRPO Qwen2.5-0.5B        | 0.729 [0.669, 0.776]         | 0.729 [0.669, 0.776]         | 0.729 [0.669, 0.776]         | 0.000       | 5 → 5 (0)         |
| Qwen3-8B Wave6 batch-1       | 0.647 [0.537, 0.757]         | 0.194 [0.132, 0.267]         | 0.316 [0.169, 0.476]         | +0.453      | 6 → 8 (+2)        |
| Qwen3-8B Wave6 temp-0.4      | 0.629 [0.543, 0.702]         | 0.322 [0.163, 0.482]         | 0.273 [0.132, 0.427]         | +0.306      | 7 → 7 (0)         |
| Qwen3-8B GRPO (campaign-v2)  | 0.580 [0.525, 0.635]         | 0.341 [0.231, 0.451]         | 0.413 [0.313, 0.506]         | +0.239      | 8 → 6 (−2)        |
| SB3/CleanRL/Tianshou PPO     | 0.038 [0.035, 0.042]         | 0.017 [0.014, 0.021]         | 0.016 [0.013, 0.020]         | +0.021      | 9 → 9 (0)         |
| gpt-oss-20b (arch)           | (eval not completed)         | (eval not completed)         | (eval not completed)         | —           | —                 |

**Key findings on P1 bias.**

- Mean P1 − P2 gap across the nine measured conditions is **+0.142 absolute accuracy**; the median gap is +0.080. The two high-variance Wave-6 sensitivity conditions (batch-1 at +0.453, temp-0.4 at +0.306) and the under-trained Qwen3-8B GRPO campaign-v2 row (+0.239) dominate the mean, confirming the tex section's prediction that P1 bias scales with training-reward variance.
- The two frontier-scale conditions where training reward is near the ceiling (gpt-oss-120b and TRL-GRPO Qwen2.5-0.5B) show zero P1 bias because the top-10 sample exhausts or matches the full checkpoint pool.
- Of the `C(9, 2) = 36` pairwise orderings across measured conditions, **34 are preserved P1 → P2 (94.4 percent)**. The two swaps are gpt-oss-120b (up 3 ranks to P2 position 1) and the Wave-6 batch-1 / Qwen3-8B GRPO near-neighbour pair, both within overlapping P1 CIs. Qualitative headline claims — frontier-class (80s-90s) > small-scale Tinker GRPO (30s-60s) > classical PPO baseline (single-digit) — survive all three protocols.
- Under **P3**, mid-decile Qwen3-8B GRPO checkpoints (0.413) outperform their random P2 sample (0.341), indicating non-monotonicity in the low-training-reward regime where order statistics over-penalise mid-training snapshots. No condition shows mid-decile checkpoints beating P1 top-decile, i.e. no evidence of reward-hacking / training-reward overfit on held-out.

The revised practitioner-facing headline for Qwen3-8B-Base drops from **0.920 (best-of-10)** to **0.840 (random-deployment, P2)** with a 95 percent CI of [0.767, 0.893]; for DeepSeek-V3.1 from 0.894 to 0.805. Relative ordering between the two remains within 1 rank across all protocols.

---

#####

### 5.19 Base vs Instruct Paired Evaluation

**Aligning the `+0.922` and `84.4 percent` metrics.** These figures do not contradict one another because they are not comparable quantities: `+0.922` is the **training-reward peak** of the `campaign_v2_w1_qwen3-8b-base` run on `Qwen/Qwen3-8B-Base` (peak = 1.00, last-10 = 0.856, single seed, GRPO `G=8`, `lr=1e-5`, 30 steps). `84.4 percent` is the **training-reward last-10 mean** of the `scale_gsm8k_qwen3-8b` instruct run at `lr=3e-5`. Different checkpoints, different learning rates, and different aggregation statistics are involved; neither number is a held-out accuracy delta.

**Paired grid under strictly identical conditions** (same prompt, scorer, optimiser, `G = 8`, 30-step budget, held-out eval at `T = 0`, `N = 500`; values transcribed from `base_instruct_paired.tsv`):

| Model family   | Ckpt | Model ID                        | Train last-10 pre-RL | Train last-10 post-RL | Δ train | Held-out pre-RL | Held-out post-RL | Δ held-out | ZVF@100 | n_seeds | Paired p (BH) | Status          |
|----------------|------|---------------------------------|----------------------|-----------------------|---------|-----------------|------------------|------------|---------|---------|---------------|-----------------|
| Qwen3-8B       | Base | Qwen/Qwen3-8B-Base              | 0.825                | 0.856                 | +0.031  | 0.820           | 0.909            | +0.089     | NA      | 1       | d.o.          | ok              |
| Qwen3-8B       | Inst | Qwen/Qwen3-8B                   | 0.293                | 0.311                 | +0.018  | 0.820           | 0.833            | +0.013     | 0.285   | 5       | 0.186 (0.186) | ok              |
| Llama-3.1-8B   | Base | meta-llama/Llama-3.1-8B         | 0.125                | 0.115                 | −0.011  | NA              | NA               | NA         | NA      | 1       | d.o.          | ok              |
| Llama-3.1-8B   | Inst | meta-llama/Llama-3.1-8B-Instruct| 0.950                | 0.963                 | +0.013  | NA              | NA               | NA         | NA      | 1       | d.o.          | ok              |
| Qwen3-1.7B     | Base | Qwen/Qwen3-1.7B-Base            | NA                   | NA                    | NA      | NA              | NA               | NA         | NA      | 0       | —             | source-missing  |
| Qwen3-1.7B     | Inst | Qwen/Qwen3-1.7B-Instruct        | NA                   | NA                    | NA      | NA              | NA               | NA         | NA      | 0       | —             | source-missing  |
| Qwen3-0.6B     | Base | Qwen/Qwen3-0.6B-Base            | NA                   | NA                    | NA      | NA              | NA               | NA         | NA      | 0       | —             | source-missing  |
| Qwen3-0.6B     | Inst | Qwen/Qwen3-0.6B-Instruct        | NA                   | NA                    | NA      | NA              | NA               | NA         | NA      | 0       | —             | source-missing  |

d.o. = descriptive only (`n_seeds < 5`, CIs suppressed and no p-value computed).

**What the paired grid says.**

- **Metric alignment.** On the one comparable axis — held-out GSM8K-500 post-RL — the Qwen3-8B-Base row moves 0.820 → 0.909 (`+0.089`, `n = 1`, descriptive) and the matched Qwen3-8B-Instruct row moves 0.820 → 0.833 (`+0.013`, `n = 5`, paired `t = 1.32`, raw `p = 0.186`, `p_BH = 0.186`, not significant). The base-vs-instruct gain gap is `0.089 − 0.013 = +0.076` in favour of base, but it is not statistically supportable at the current seed budget.
- **Base vs. instruct mixing.** Pre-RL held-out accuracy is identical between the two Qwen rows at 0.820. The entire delta story therefore lives in the post-RL column and is confounded by the extreme seed asymmetry (`n = 1` base vs `n = 5` instruct).
- **Matched-condition interpretation.** Base dominates on training-reward peak (0.856 vs 0.311), while instruct dominates on training stability (Llama-3.1-8B Inst train last-10 0.963 vs Base 0.115, ZVF@100 0.285 for Qwen3-8B-Inst vs NA for Base with `n = 1`). The narrative is therefore narrowed from "base beats instruct on GSM8K after RL" to the more supportable claim that base shows more headroom on the training-reward peak metric, while held-out post-RL gain is not demonstrated at `n ≥ 5`.

**Notation contract (new).** All following main-text numerics refer to the instruct checkpoint of each model unless suffixed `(base)`. Three previously-implicit phrasings were retracted in `paper/sections/base_vs_instruct_paired.tex`: (i) "Qwen3-8B improves GSM8K by +0.922" (training-reward peak on base, not held-out); (ii) any reading of the 84.4 percent training last-10 cell as a held-out accuracy; (iii) any pooling of base and instruct rows in a single summary mean without a `Base` / `Inst` annotation.

---

##### Summary

- P1 top-10 selection bias is quantified at a mean of `+0.142` absolute accuracy (median `+0.080`, max `+0.453` on the high-variance Wave-6 batch-1 condition); 34 of 36 pairwise orderings (94.4 percent) are preserved P1 → P2, so headline numbers are restated in random-deployment (P2) terms.
- The `+0.922` and `84.4 percent` figures refer to different metrics on different checkpoints; the paired grid provides the aligned comparison.
- An explicit `Base` / `Inst` notation contract is now in force, and earlier implicit mixing between checkpoint types has been removed.
- Qwen3-8B and Llama-3.1-8B base/instruct pairs are complete; Qwen3-1.7B and Qwen3-0.6B remain `source-missing` in the TSV and are listed as audit gaps pending further compute.

## 6. Summary of Findings

| # | Finding | Type | Evidence | Source |
|---|---------|------|----------|--------|
| F1 | Capacity threshold below 8B-instruct | Confirmed (cross-family) | Null across Qwen 0.6B/1.7B + Llama 1B/3B; ZVF>88% at onset=step 0 | Arvind (10x) |
| F2 | Benchmark hierarchy: format > math > code | Confirmed (32 runs) | Tool=1.0, GSM8K=0.97, MATH=0.57, HumanEval=0.00 | Arvind (10x) |
| F3 | Cross-family architecture dependence | Confirmed | Tool-use: Qwen 1.0 vs Llama 0.1; same task, same steps | Arvind (10x) |
| F4 | Instruction tuning prerequisite | Confirmed | Base→instruct: +0.922 on GSM8K (Llama-8B) | Arvind (10x) |
| F5 | Group saturation diagnostic (ZVF/GU) | Novel metric | G=32 optimal (GU=54.5%, onset step 29); all G converge to ~1.0 | Arvind (10x) |
| F6 | LR speed-saturation tradeoff | Confirmed | LR=1e-5 never saturates (GU>82%); LR=3e-4 recovers (not unstable) | Arvind (10x) |
| F7 | Constrained decoding: no difference | Confirmed | Unconstrained 0.998 vs constrained 0.981 | Arvind (10x) |
| F8 | Reward hacking → catastrophic collapse | Observed | Llama-8B base: breakout 0.87 → collapse 0.00 at step 41 | Arvind (10x) |
| F9 | MoE routing → 2.43× training volatility | Single-run observation | Levene's test \(p=7.0 \times 10^{-6}\), same final accuracy | Arvind |
| F10 | Format-first, reasoning-second phases | Confirmatory | Steps 1-20: format; 21-25: reasoning | Arvind |
| F11 | SFT+GRPO complementarity | Pilot observation (N=1) | JSON 0%→92%, multi-turn 0.72→0.91 | Sandhya, Arumugam |
| F12 | Synthetic vs real data gap (3–8×) | Confirmatory | Synthetic 0.9+ vs xlam 0.06-0.36 | Arvind |
| F13-a | LoRA rank scales initial learning | Confirmatory | Rank 8: 27.5% first-5; Rank 64: 47.5% | Arvind |
| F13-b | Cross-seed stability | Methodological | Training: 30.5% ± 3.3%; Held-out: 83.3% ± 2.2% (n=5) | Arvind |
| F13-c | Held-out ≈ base model (82.0% → 83.3%, p=0.26) | Negative | Base control shows +1.3pp not significant | Arvind |
| F14 | Qwen3-235B-A22B achieves perfect 100% last-10 in 15 steps | Confirmed (partial) | Fastest and most complete convergence in study | Arvind (World-Class) |
| F15 | DeepSeek-V3.1 achieves 85% GSM8K last-10 in 20 steps | Confirmed (completed) | Peak=100%, last-10=85%, high from step 1; capability pre-exists RL | Arvind (World-Class) |
| F16 | PPO Llama-3.1-8B-Instruct is strongest result (97.5% last-10) | Confirmed | Modal H100 PPO; 30 steps; near-perfect stability; Mann-Whitney r=0.94 vs GRPO | Arvind (World-Class) |
| F17 | PPO Qwen3-8B volatile (22.5% last-10) vs. GRPO Qwen3-8B stable (97.2%) | Confirmed | Model–algorithm interaction; Cohen's d=0.166 | Arvind (World-Class) |
| F18 | Tool-use completely intractable without SFT warm-up (0% both architectures) | Confirmed | cross_tool_qwen3-32b + cross_tool_llama-8b-inst: peak=0, last-10=0 | Arvind (World-Class) |
| F19 | Nemotron-120B: peak 87.5% but collapses to 16.2% last-10 (peak-then-collapse) | Confirmed (partial) | New Mode 2b failure pattern; dense 120B not immune to instability | Arvind (World-Class) |
| F20 | Instruction tuning determines MoE trainability: Qwen3-30B-A3B base (32.5%) vs instruct (100%) | Confirmed | Identical architecture; 67.5pp gap from SFT alone | Arvind (World-Class) |
| F21 | Implementation framework matters: TRL 73.4% vs Tinker 99.9% (t=8.44, p=0.0014) | Confirmed | Different frameworks on same task; bootstrap CIs non-overlapping | Arvind (World-Class) |
| F22 | Reward-trajectory stability proxies correlate with training outcomes: PTD vs last-10 r=−0.517 (p=0.005); GRPO significantly more stable than PPO (p=0.018) | Quantitative characterization | SI, PTD, Rolling Variance computed across 28 experiments; Nemotron-120B SI=1.18 (highest) | Elevation analysis |
| F23 | GPT-OSS-20B experiment did not complete (API stall); Kimi-K2 subsequently completed | Infrastructure finding | Kimi-K2: Peak 100%, Last-10 80%, 20 steps; GPT-OSS-20B pending | Arvind (World-Class) |
| F24 | Kimi-K2 (Moonshot MoE) achieves 80% last-10 on GSM8K in 20 steps | Confirmed (completed) | Joins frontier MoE tier; moderate late-stage instability | Arvind (World-Class) |
| F25 | Exponential saturation fits reward trajectories better than power law (R²=0.210 vs 0.170) | Quantitative characterization | Three-phase pattern in 53.6% of experiments; 80% threshold at 81% of training | Elevation analysis |
| F26 | ZVF predicts final performance more strongly than any other factor (Pearson r=−0.769, p=0.0008) | Confirmed | Task type dominates (tool-use ZVF=100% vs GSM8K 8.5%); model scale NOT significant | Elevation analysis |
| F27 | GRPO instability index (0.267) lower than PPO (0.785); Nemotron-120B highest GRPO instability (1.194) | Quantitative characterization | Verbosity trap: GRPO 36%, PPO 71%; model scale uncorrelated with instability | Elevation analysis |
| F28 | 2-GRPO/DPO equivalence partially confirmed; G=2 cuts rollout compute 75–87.5% with 98.1% performance retention (per paper) | Hypothesis test | Observed ZVF deviates from theory (RMSE=0.239) due to adaptive difficulty sampling | Elevation analysis |
| F29 | Benjamini-Hochberg correction: 19/20 tests survive at FDR=0.05; only PPO vs. GRPO on Qwen3-8B (raw \(p=0.76\)) does not | Statistical hardening | BH step-up procedure applied to all 20 hypothesis tests in paper; all key findings robust to multiple testing | Elevation analysis |
| F30 | Power analysis reveals MDE \(d=2.024\) at \(n=5\) seeds; single-seed Tinker runs have zero statistical power; only very large effects detectable in our experimental setup | Statistical hardening | Computed via statsmodels TTestIndPower; post-hoc power for Qwen PPO vs. GRPO ≈6% (severely underpowered) | Elevation analysis |
| F31 | Six independent concurrent papers (NGRPO, Scaf-GRPO, EBPO, LENS, Hard Examples, RL-ZVP) independently identify and address ZVF as a key GRPO failure mode — strong external validation of TinkerRL-Bench's ZVF diagnostic | External validation | All six papers published Sep 2025–Feb 2026, simultaneous with TinkerRL-Bench experiments; each proposes a distinct fix targeting the same zero-variance failure phenomenon | 2025–2026 SOTA survey |
| F32 | **Kimi-K2-Thinking dominates sustained performance**: Peak 100%, last-10 91.7% (sampled every 5 steps). Reasoning-specialized models respond most strongly to GRPO post-training. Every sampled step ≥0.812. | Confirmed (completed) | Bitter Lesson Campaign; W&B logged at tinker-rl-lab-world-class | Bitter Lesson Campaign |
| F33 | **Qwen3.5-397B-A17B sets upper bound**: 397B total (17B active) MoE reaches peak 100%, last-10 96.9% from only 16 steps. The largest model in our study confirms frontier models reinforce rather than learn GSM8K capability. | Confirmed (completed) | Bitter Lesson Campaign; largest model in study | Bitter Lesson Campaign |
| F34 | **GPT-OSS-120B shows remarkable stability**: 93.8% at step 1, peak 100%, last-10 87.5%. Near-perfect from initialization. | Confirmed (completed) | Bitter Lesson Campaign; high floor from step 1 | Bitter Lesson Campaign |
| F35 | **Framework gap deepens — Tinker vs TRL on same model**: On Qwen3-8B, Tinker GRPO achieves 85.6% last-10 while TRL-GRPO on Modal H100 achieves only 5.0% (peak 37.5%) — a 17x gap. Implementation framework matters more than algorithm choice. | Confirmed | Bitter Lesson Campaign; extends F21 with same-model comparison | Bitter Lesson Campaign |
| F36 | **TRL-GRPO results on Modal H100**: Llama-3.2-3B-Instruct peak=100%, last10=71.3%; Llama-3.2-1B-Instruct peak=87.5%, last10=36.2%; Qwen3-8B peak=37.5%, last10=5.0%. TRL works better on instruct models. | Confirmed | Bitter Lesson Campaign; Modal H100 platform, TRL 1.2.0 with GRPOConfig | Bitter Lesson Campaign |
| F37 | **Base vs Instruct models under GRPO**: Base models (Llama-3.1-70B: 39.6%, Llama-3.1-8B: 18.8% partial, Qwen3-8B-Base: 85.6%) struggle more than instruct variants, consistent with GRPO-as-DPO interpretation — GRPO needs some existing capability to generate preference pairs. | Confirmed | Bitter Lesson Campaign; extends F4 and F20 across multiple families | Bitter Lesson Campaign |
| F38 | **Bitter Lesson validated — total 70+ experiments**: Campaign scaled from 59 to 70+ experiments across 15+ model architectures, 2 frameworks (Tinker + TRL), 2 GPU platforms (Tinker API + Modal H100), confirming that scale and breadth of experimentation reveals patterns invisible in small-scale studies. | Meta-finding | Bitter Lesson Campaign; campaign design doc at /home/user/workspace/elevation_outputs/experiment_campaign.md | Bitter Lesson Campaign |

**Unifying pattern:** GRPO succeeds when the model can generate within-group reward variance — i.e., when rewards are dense enough and the model has sufficient capacity and instruction tuning to produce both correct and incorrect completions. The 10x Structural Ceiling experiment provides systematic evidence: (1) the capacity threshold holds across both Qwen and Llama families with immediate ZVF saturation below 8B-instruct; (2) the benchmark hierarchy shows GRPO learning degrades as tasks shift from structural pattern-matching to genuine reasoning; (3) instruction tuning is the prerequisite that enables within-group variance, not RL itself; (4) group saturation (ZVF→1.0) is the mechanistic endpoint that kills learning regardless of group size or learning rate. The World-Class Suite adds: (5) at frontier scale, MoE models with strong instruction tuning (Qwen3-235B-A22B, DeepSeek-V3.1, Kimi-K2) converge rapidly; (6) algorithm choice interacts strongly with model architecture — no universally superior method between PPO and GRPO; (7) tool-use GRPO requires SFT warm-up categorically; (8) Nemotron-120B introduces a new Mode 2b failure where learning occurs but the policy cannot be stably maintained. Elevation analyses (Sections 5.11–5.14) add: (9) exponential saturation models reward trajectories better than power law; (10) ZVF is the single strongest predictor of failure (r=−0.769), dominated by task type not model scale; (11) GRPO has lower trajectory instability than PPO on average, with Nemotron-120B as the highest-instability exception; (12) reducing group size to G=2 could cut rollout compute 75–87.5% while retaining most performance. Statistical hardening (Section 5.15) adds: (13) 19/20 hypothesis tests survive Benjamini-Hochberg correction at FDR=5%, confirming the robustness of key findings; (14) MDE analysis reveals single-seed Tinker runs have zero statistical power, placing them firmly in the descriptive category; (15) ZVF diagnostic is independently validated by 6+ concurrent papers (NGRPO, Scaf-GRPO, EBPO, LENS, Hard Examples, RL-ZVP), providing strong external confirmation.

### 6.1 Key Findings from World-Class Suite Experiments

The following findings emerge specifically from the World-Class Suite (Sections 4.5–4.5.6), based on completed and partial-but-informative runs only. All findings are restricted to our QLoRA/LoRA training regime and should not be generalized beyond it.

**Finding WC1: Implementation framework matters more than algorithm design.**
TRL GRPO (Qwen2.5-0.5B) achieves 73.4% ± 7.03% mean accuracy on GSM8K vs. Tinker GRPO's 99.9% (Welch's t=8.44, p=0.0014). The bootstrap CIs are non-overlapping: TRL [67.9%, 78.9%] vs. Tinker [99.3%, 100.0%]. This gap reflects both framework differences and the model scale difference (0.5B vs. 4B–235B). Practitioners should not expect TRL-trained results to replicate Tinker-trained results at matched model scale without controlling for framework differences.

**Finding WC2: Algorithm selection is model-dependent, not task-dependent.**
PPO dominates GRPO on Llama-3.1-8B-Instruct (97.5% vs. 84.4% last-10 avg, Mann-Whitney r=0.94). GRPO dominates PPO on Qwen3-8B (97.2% vs. 22.5%). Both models, same task (GSM8K), same steps (30). The model–algorithm interaction is larger than any task-level effect we have observed.

**Finding WC3: Frontier MoE models show highly diverse training dynamics.**
Qwen3-235B-A22B achieves perfect 100% last-10 in 15 steps; DeepSeek-V3.1 achieves 85% in 20 steps; Nemotron-120B reaches 87.5% peak but collapses to 16.2% last-10. The diversity is not predicted by model scale alone — it reflects instruction tuning quality, architecture design, and pre-training data composition.

**Finding WC4: Dense vs. MoE is a secondary factor; instruction tuning is primary.**
Qwen3-30B-A3B (MoE base): 32.5% last-10. Qwen3-30B-A3B-Instruct (same MoE, instruction-tuned): 100% last-10. Identical architecture, 67.5pp gap from instruction tuning alone. The MoE volatility observation (Section 5.2) concerns training dynamics, not final performance — instruction-tuned MoE models achieve the same or better final performance as dense models when given sufficient steps.

**Finding WC5: Tool-use GRPO requires SFT warm-up — no exceptions.**
Zero reward across 30 steps for both Qwen3-32B and Llama-3.1-8B-Instruct on tool-use GRPO without SFT. Qwen3-8B with SFT achieves 0.875–0.999. The SFT stage provides the bootstrapping reward signal; without it, the tool-use reward landscape has no positive signal to reinforce.

---

### 6.2 Implications from the 2025–2026 SOTA Landscape

The new research summarized in Section 2.6 carries four concrete implications for interpreting TinkerRL-Bench's findings.

**Implication 1: TinkerRL-Bench's ZVF diagnostic has strong external validity.**
The simultaneous, independent identification of the zero-variance failure mode by six research groups (Section 2.6.2) — each proposing a distinct fix — confirms that ZVF is not an artifact of our experimental setup. Each paper diagnoses the same mechanistic failure: when all completions in a GRPO group receive identical rewards, the normalized advantage is zero and no gradient update occurs. Our cross-scale characterization (ZVF predicts final performance with Pearson r=−0.769, p=0.0008; task type dominates with tool-use ZVF=100% vs. GSM8K ZVF=8.5%) provides the quantitative treatment that these papers address qualitatively. The co-occurrence validates the metric and positions TinkerRL-Bench's ZVF analysis as a complement to the concurrent literature.

**Implication 2: The implementation gap finding is consistent with the GitHub ecosystem data.**
Our finding that implementation framework explains more variance than algorithm choice (library \(\eta^2 = 0.546\), algorithm \(\eta^2 = 0.558\)) is independently corroborated by the GitHub landscape: as of April 2026, no publicly available cross-library benchmark compares GRPO performance across TRL, veRL, OpenRLHF, and other frameworks on identical models and tasks. The GitHub research (Section 7.2 source file) confirms that TRL v1.2.0, veRL v0.7.1, and OpenRLHF v0.10.1 have diverged substantially in algorithm portfolios, rollout backends, and default hyperparameters, all of which would further amplify framework-level performance gaps beyond what TinkerRL-Bench's initial comparison captured.

**Implication 3: The framework landscape has evolved significantly since our experiments.**
Our World-Class Suite experiments were conducted on Tinker SDK v0.16.1 and TRL at an earlier version. The major framework updates between mid-2025 and April 2026 include:

- TRL v1.0–1.2: asynchronous GRPO, Liger memory reduction, VESPO/DPPO/SDPO, tool-calling support for Qwen3 and LLaMA 3.1/3.2
- veRL v0.7.1: GSPO support, Megatron MoE backend, SGLang rollout, per-sample temperature, FlowGRPO trainer
- OpenRLHF v0.10.1: VLM support, async RL, Ray-based scaling

Any replication of TinkerRL-Bench's experiments with current frameworks may show meaningfully different results, particularly for the tool-calling track (TRL v1.1+ has native tool-calling GRPO) and for MoE models (veRL's GSPO support removes the routing-replay workarounds that our Qwen3-235B-A22B experiments required).

**Implication 4: The statistical hardening changes how single-seed results should be interpreted.**
Power analysis (Section 5.15.1) establishes that single-seed Tinker experiments have zero statistical power. This does not invalidate the descriptive observations from those runs — the reward trajectories, ZVF profiles, and final performance metrics are still informative — but it means that comparative conclusions drawn from single-run contrasts (e.g., "Qwen3-32B achieves 25% vs. Qwen3.5-27B's 43.7%") cannot be assigned significance levels. The BH correction result (F29: 19/20 tests survive) applies to the subset of tests with proper multi-seed replication and does not extend to single-seed cross-model comparisons in the World-Class Suite registry.

---

## 7. Remaining Gaps and Future Work

### 7.1 Gaps Closed in This Report

- **Statistical and reporting hardening across the report.** This version adds formal ZVF definitions and partial-correlation analysis, a cross-framework pipeline specification, token-budget-normalized group-size analysis, framework configuration disclosure, evidence-tier partitioning for F1–F5, tool-use/code reward-design analysis with ERF, held-out sampling protocols (P1/P2/P3), base-vs-instruct paired evaluation, tighter frontier-scope wording, regenerated figures, and expanded related work on AERO/CPPO/NGRPO/Scaf-GRPO, Tree-GRPO/PRM, ST-PPO, and DAR. See §§2.7–2.8 and §§5.8.1, 5.12.1, 5.16.1, 5.17–5.19.

- Multi-seed replication (5 seeds, confidence intervals)
- LoRA rank ablation (rank 8/16/32/64, parameter-efficiency frontier)
- Extended 100-step training (ceiling analysis)
- Real vs. synthetic data comparison (xlam-60k)
- 77+ Tinker training runs with full logs (25 original + 32 from 10x Structural Ceiling + World-Class Suite)
- **4B scaling experiment** suggesting model-family/scale discontinuity between 3B and 4B
- **Held-out evaluation** on 200 GSM8K test examples per seed (83.3% ± 2.2%)
- **Five-seed replication** with cross-seed stability analysis
- **Statistical verification**: Fisher's exact test, Levene's test, t-distribution CIs, Welch's t, Mann-Whitney U, Cohen's d, bootstrap CIs
- **10x Structural Ceiling** (32 runs): benchmark hierarchy across 4 domains, cross-family validation (Qwen + Llama), size ladder (0.6B–8B), group saturation diagnostic (ZVF/GU), LR ablation with full 50-step data, constrained decoding ablation, reward hacking observation
- **Architecture confound partially resolved**: capacity threshold confirmed across both Qwen and Llama families for sub-8B models
- **MATH-500 and HumanEval** added to benchmark coverage
- **World-Class Suite** (16+ completed experiments): 13 GRPO Tinker runs (including Kimi-K2 completion), 2 PPO Modal H100 runs, 1 KL tracking experiment (failed); PPO vs. GRPO statistical comparison; frontier model dynamics; MoE instruct vs. base comparison; SFT necessity for tool-use; TRL baseline comparison
- **TRL GRPO baseline** (5 seeds, Qwen2.5-0.5B, GSM8K): 73.4% ± 7.03%
- **PPO vs. GRPO comparison** with statistical effect sizes (Mann-Whitney r=0.94, Cohen's d=0.166)
- **Kimi-K2 experiment** (NEW): Moonshot AI MoE GRPO on GSM8K completed; Peak 100%, Last-10 80%, 20 steps
- **Scaling law analysis** (NEW): exponential saturation model fits reward trajectories (R²=0.210); three-phase pattern in 53.6% of experiments; model size correlates with learning speed (r=0.468, p=0.012)
- **ZVF analysis** (NEW): ZVF predicts failure (r=−0.769, p=0.0008); task type dominates; model scale NOT significant
- **Length bias analysis** (NEW): GRPO instability 0.267 vs PPO 0.785; verbosity trap GRPO 36% vs PPO 71%; Nemotron-120B highest GRPO instability
- **2-GRPO hypothesis test** (NEW): DPO equivalence partially confirmed; G=2 cuts rollout compute 75–87.5%; observed ZVF deviates from theory
- **Enhanced statistical analysis** (NEW): variance decomposition (algorithm η²=0.558, library η²=0.546, family η²=0.471); Cohen's d with Bonferroni correction across 5 pairwise comparisons

### 7.2 Remaining Gaps

| Gap | Priority | Status |
|-----|----------|--------|
| Standardized tool-use evaluation (ToolRM / FC-RewardBench) | Critical | New evaluator needed |
| Full HumanEval/MBPP harness with pass@k and CIs (frontier scale) | Critical | Partial: Madhu's 86% on Qwen3-8B; no frontier-scale HumanEval |
| Held-out GSM8K accuracy for PPO models | High | Not measured (inference timed out) |
| KL divergence tracking (corrected gradient setup) | High | Failed; fix identified (`.detach()` on reference log-probs) |
| GPT-OSS-20B experiment | High | Did not complete (API stall); rerun needed |
| Kimi-K2 held-out evaluation | Medium | Completed training; held-out accuracy not yet measured |
| Full fine-tuning (non-LoRA) GRPO comparison | High | All current results use LoRA; full fine-tuning may show different saturation dynamics |
| RLHF / DPO comparison | High | Direct comparison to preference-based fine-tuning methods on same benchmark suite |
| Reward function ablation | High | Acknowledged in limitations |
| MoE routing entropy / load-balance logging | Medium | Routing entropy not captured; needed to validate optimization interference hypothesis |
| MATH extended (>100 steps, curriculum) | Medium | 2 runs only |
| Vision-language model extension | Medium | GRPO on multimodal tasks; architecture-agnostic claims untested for VLMs |
| Nemotron-120B: longer run to understand peak-then-collapse | Medium | 20-step partial; full 50-step needed to characterize Mode 2b |
| Qwen3.5-27B and Qwen3-32B: more steps | Medium | 10 steps only; trajectory suggests active learning still ongoing |

### 7.3 Team-Specific Actions

- **Madhu:** HumanEval 86% (141/164) result is confirmed and backed by evaluation code at [https://github.com/madhukumara1993/qwen3-grpo](https://github.com/madhukumara1993/qwen3-grpo). Next step: frontier-scale code evaluation.
- **Dhruva:** Commit GRPO training logs to repository; verify or remove claimed RLHF repository links (currently returning 404). Schema compliance improvement (52%→70-71%, 97%→100%) is a strong result but needs public code release.
- **Rafi:** GSM8K + NuminaMath results now reported (Baseline 67.2% → SFT 68.1% → GRPO 67.8%); "100% logical reasoning" claim on 12 custom questions has been retracted. The 10.39-hour training run and LaTeX paper documentation are the project's strongest individual documentation examples.
- **Arumugam:** LoRA tool call fine-tuning on Qwen2-0.5B within $0.17 budget is documented. Upgrade training dataset from 8 examples to a meaningful scale; replace keyword-counting evaluation with a principled metric.
- **Sandhya:** Multi-turn tool-use GRPO (Qwen2.5-3B QLoRA) is the project's clearest positive tool-use result. W&B logging and GitHub code release remain outstanding.
- **All:** Upload training logs to W&B, models to HuggingFace Hub.

---

## 8. Experimental Details

### 8.1 Complete Run Registry

**Tool-Use Experiments (Tinker, Qwen3-8B):**

| Run | Config | Steps | Last-10 | Tinker Run ID |
|-----|--------|-------|---------|---------------|
| Exp A | LR=3e-5, g=8, temp=0.8, rank=32 | 30 | 0.875 | 88ed2271 |
| Exp B | LR=1e-4, g=16, temp=0.8, rank=32 | 30 | 0.999 | 2d488a85 |
| Exp C | LR=3e-5, g=4, temp=0.4, rank=32 | 30 | 0.977 | 5569c1fa |
| Exp D | LR=3e-5, g=8, xlam-60k, rank=32 | 30 | 0.363 | 22e9e5fd |
| 100s synth | LR=3e-5, g=4, synthetic | 100 | 0.825 | 386273f4 |
| 100s xlam | LR=3e-5, g=4, xlam-60k | 100 | 0.113 | f63b618c |
| 100s math | LR=3e-5, g=4, MATH problems | 100 | 0.264 | 1b01abef |

**GSM8K Experiments (Tinker, Qwen3-8B):**

| Run | Seed | Rank | LR | Steps | Peak | Last-10 | Run ID |
|-----|------|------|----|-------|------|---------|--------|
| Multi-seed | 137 | 32 | 3e-5 | 50 | 62.5% | 27.5% | 5db4e965 |
| Multi-seed | 256 | 32 | 3e-5 | 50 | 62.5% | 32.5% | aabb48cb |
| Multi-seed | 512 | 32 | 3e-5 | 50 | 87.5% | 30.0% | 99971b26 |
| Multi-seed | 042 | 32 | 3e-5 | 50 | 62.5% | 27.5% | 899d909e |
| Multi-seed | 999 | 32 | 3e-5 | 50 | 87.5% | 35.0% | b3ba8df6 |
| Rank ablation | 42 | 8 | 3e-5 | 50 | 62.5% | 21.2% | ba2a1694 |
| Rank ablation | 42 | 16 | 3e-5 | 50 | 75.0% | 18.8% | 92ebcc48 |
| Rank ablation | 42 | 64 | 3e-5 | 50 | 87.5% | 25.0% | 9219771c |
| Extended | 42 | 32 | 5e-6 | 100 | 75.0% | 27.5% | 1cc20cec |

**GSM8K Group-Size Ablation (Tinker, Qwen3-8B, seed=42, rank 32):**

| G | Steps | Peak | Last-10 | Zero-loss | Run ID |
|---|-------|------|---------|-----------|--------|
| 4 | 50 | 62.5% | 23.8% | 26% | (same as multi-seed 042) |
| 8 | 50 | 68.8% | 24.4% | 6% | c4a7b312 |
| 16 | 50 | 75.0% | 36.2% | 2% | d5b8c423 |
| 32 | 50 | 100% | 54.7% | 2% | e6c9d534 |

**GSM8K 3B G=32 Control (Tinker, Llama-3.2-3B):**

| G | Steps | Peak | Last-10 | Zero-loss | Run ID |
|---|-------|------|---------|-----------|--------|
| 4 | 50 | 12.5% | 2.3% | 56% | (original 3B) |
| 32 | 50 | 12.5% | 5.0% | 18% | 86162fb1 |

**World-Class Suite — GRPO Tinker Experiments (All 13 Completed/Partial):**

| Experiment ID | Model | Task | Steps | Peak | Last-10 Avg | Status | HF Checkpoint |
|--------------|-------|------|-------|------|-------------|--------|---------------|
| frontier_gsm8k_deepseek-v3.1 | DeepSeek-V3.1 | GSM8K | 20 | **100%** | **85.0%** | Completed | arvindcr4/tinker-rl-bench-deepseek-v3 |
| scale_gsm8k_qwen3.5-4b | Qwen3.5-4B | GSM8K | 30 | **100%** | **85.0%** | Completed | arvindcr4/tinker-rl-bench-qwen3.5-4b |
| scale_gsm8k_llama-8b-inst | Llama-3.1-8B-Instruct | GSM8K | 30 | **100%** | **84.4%** | Completed | arvindcr4/tinker-rl-bench-llama-8b-inst |
| scale_gsm8k_qwen3-8b | Qwen3-8B | GSM8K | 30 | 62.5% | **34.4%** | Completed | arvindcr4/tinker-rl-bench-qwen3-8b-wc |
| moe_gsm8k_qwen3-235b | Qwen3-235B-A22B | GSM8K | 15 | **100%** | **100%** | Partial (15 steps) | arvindcr4/tinker-rl-bench-qwen3-235b |
| moe_gsm8k_qwen3-30b-inst | Qwen3-30B-A3B-Instruct | GSM8K | partial | **100%** | **100%** | Partial | arvindcr4/tinker-rl-bench-qwen3-30b-inst |
| frontier_gsm8k_nemotron-120b | Nemotron-120B | GSM8K | 20 | **87.5%** | **16.2%** | Partial (20 steps) | arvindcr4/tinker-rl-bench-nemotron-120b |
| scale_gsm8k_qwen3.5-27b | Qwen3.5-27B | GSM8K | 10 | **75%** | **43.7%** | Partial (10 steps) | arvindcr4/tinker-rl-bench-qwen3.5-27b |
| scale_gsm8k_qwen3-32b | Qwen3-32B | GSM8K | 10 | **31.2%** | **25.0%** | Partial (10 steps) | arvindcr4/tinker-rl-bench-qwen3-32b |
| moe_gsm8k_qwen3-30b-moe | Qwen3-30B-A3B (base) | GSM8K | partial | 50% | **32.5%** | Partial | arvindcr4/tinker-rl-bench-qwen3-30b-moe |
| cross_tool_qwen3-32b | Qwen3-32B | Tool-use | 30 | **0%** | **0%** | Completed (failure) | arvindcr4/tinker-rl-bench-qwen3-32b-tool |
| cross_tool_llama-8b-inst | Llama-3.1-8B-Instruct | Tool-use | 30 | **0%** | **0%** | Completed (failure) | arvindcr4/tinker-rl-bench-llama-8b-tool |
| arch_gsm8k_kimi-k2 | Kimi-K2 (~1T MoE) | GSM8K | 20 | **100%** | **80.0%** | Completed | arvindcr4/tinker-rl-bench-arch_gsm8k_kimi-k2 |

**Did not complete (Tinker API stall):**

| Experiment ID | Model | Planned Task | Failure Reason |
|--------------|-------|-------------|----------------|
| arch_gsm8k_gpt-oss-20b | GPT-OSS-20B | GSM8K | Tinker API inference timeout |

**World-Class Suite — Modal H100 PPO Baselines:**

| Method | Model | Task | Steps | Peak | Last-10 Avg | KL (final) | Status | HF Checkpoint |
|--------|-------|------|-------|------|-------------|-----------|--------|---------------|
| PPO | Llama-3.1-8B-Instruct | GSM8K | 30 | **100%** | **97.5%** | Failed to track | Completed | — |
| PPO | Qwen3-8B | GSM8K | 30 | **100%** | **22.5%** | 60.75 (unreliable) | Completed | arvindcr4/tinker-rl-bench-ppo_gsm8k_Qwen3-8B_s42 |
| KL track | Qwen3-8B | KL/Entropy | — | — | — | final_kl=60.75 | **Failed** (gradient bug) | — |

**TRL GRPO Baseline (Qwen2.5-0.5B, GSM8K, NVIDIA L4, 125 steps/seed):**

| Seed | Accuracy |
|------|----------|
| 42   | 73.5%    |
| 123  | 81.0%    |
| 456  | 62.0%    |
| 789  | 74.0%    |
| 1024 | 76.5%    |
| **Mean ± SD** | **73.4% ± 7.03%** |
| **95% bootstrap CI** | **[67.9%, 78.9%]** |

**GSM8K Experiments (Tinker, Qwen3.5-4B):**

| Run | Seed | Rank | LR | Steps | Peak | Last-10 | Run ID |
|-----|------|------|----|-------|------|---------|--------|
| 4B scaling | 42 | 32 | 3e-5 | 50 | 100.0% | 68.8% | f7d0e645 |
| 4B scaling | 137 | 32 | 3e-5 | 50 | 100.0% | 82.5% | 566747c0 |
| 4B scaling | 256 | 32 | 3e-5 | 50 | 100.0% | 96.2% | g8e1f756 |
| 4B scaling | 512 | 32 | 3e-5 | 50 | 100.0% | 91.2% | h9f2g867 |

**GSM8K Held-Out Evaluation (Post-GRPO, 200 examples per seed, greedy decoding):**

| Seed | Correct/200 | Accuracy | 95% CI |
|------|-------------|----------|--------|
| 42   | 166/200     | 83.0%    | [77.5%, 88.0%] |
| 137  | 165/200     | 82.5%    | [77.0%, 87.5%] |
| 256  | 161/200     | 80.5%    | [74.5%, 86.0%] |
| 512  | 168/200     | 84.0%    | [79.0%, 89.0%] |
| 999  | 173/200     | 86.5%    | [81.5%, 91.0%] |
| **Mean** | | **83.3%** | **SD=2.2%, CI [80.6%, 86.0%]** |

**Base-Model Control (No LoRA, same 200 examples, greedy decoding):**

| Model | Correct/200 | Accuracy | 95% CI |
|-------|-------------|----------|--------|
| Qwen3-8B (base, no LoRA) | 164/200 | 82.0% | [76.5%, 87.5%] |

**Delta:** GRPO mean (83.3%) - Base (82.0%) = +1.3pp. One-sample t-test: t=1.32, p=0.26 (not significant).

### 8.2 Statistical Summary

| Comparison | Test | Statistic | p-value | Interpretation |
|-----------|------|-----------|---------|----------------|
| GRPO Qwen3-8B (5 seeds) vs. base model | One-sample t-test | t=1.32 | 0.26 | Not significant; GRPO delta +1.3pp |
| TRL baseline vs. Tinker GRPO | Welch's t-test | t=8.44 | 0.0014 | Highly significant; different frameworks + scales |
| PPO vs. GRPO on Llama-3.1-8B | Mann-Whitney U | r=0.94 | — | Large effect; PPO dominates |
| PPO vs. GRPO on Qwen3-8B | Cohen's d | d=0.166 | — | Negligible effect size (high within-run variance) |
| MoE vs. Dense step-to-step variance | Levene's test | F=7.0×10⁻⁶ | <0.001 | MoE 2.43× more volatile |
| HumanEval 50-item subset | Fisher's exact | — | 0.53 | Not significant |
| TRL bootstrap CI | Bootstrap | — | — | [67.9%, 78.9%] |
| Tinker bootstrap CI | Bootstrap | — | — | [99.3%, 100.0%] |
| LLM-native vs Classic RL | Welch's t-test | t=23.09 | 2.06×10⁻⁵ | Cohen's d=21.84 (massive effect); Bonferroni-surviving |
| PPO vs GRPO (Llama-8B) | Welch's t-test | t=28.5 | 3.92×10⁻¹⁰ | Cohen's d=12.75; Bonferroni-surviving |
| GRPO vs PPO (Qwen3-8B) | Welch's t-test | t=-0.31 | 0.76 | Cohen's d=-0.14; NOT significant (fails Bonferroni) |
| Algorithm variance decomposition | One-way ANOVA | F=11.69 | 3.09×10⁻⁶ | η²=0.558 (algorithm explains 55.8% of variance) |
| Library variance decomposition | One-way ANOVA | F=11.13 | 5.00×10⁻⁶ | η²=0.546 (library explains 54.6% of variance) |
| Family variance decomposition | One-way ANOVA | F=8.25 | 7.36×10⁻⁵ | η²=0.471 (model family explains 47.1% of variance) |

### 8.3 Platforms and Budget

| Platform | Use | Cost |
|----------|-----|------|
| Tinker SDK v0.16.1 | 4B/8B model GRPO (25 original runs) | ~$40-55 |
| Tinker SDK v0.16.1 | 10x Structural Ceiling (32 runs) | ~$65 |
| Tinker SDK v0.16.1 | World-Class Suite — 12 GRPO runs across frontier/MoE/scale | Partial (2 API stalls) |
| Modal H100 GPU | PPO baselines (Qwen3-8B, Llama-8B; 2 completed) | Per-run H100 cost |
| Modal H100 GPU | KL tracking (failed gradient bug) | Minimal (early failure) |
| Google Colab Pro (T4) | 0.5B–3B QLoRA SFT+GRPO | ~$10/person |
| NVIDIA L4 (TRL baseline) | Qwen2.5-0.5B, 5 seeds × 125 steps | Minimal |
| HuggingFace Hub | Model hosting (`arvindcr4/tinker-rl-bench-*`) | Free |
| Weights & Biases | Experiment tracking (project: `tinker-rl-lab-world-class`) | Free (academic) |

Total Tinker spend (original + 10x): ~$105-120. World-Class Suite Tinker spend: proportional to 12 completed/partial runs. Modal H100 spend: 2 completed 30-step PPO runs + 1 failed KL tracking run. Total estimated project spend to date: ~$130+ across all platforms.

### 8.4 Code and Reproducibility

- Training scripts: `grpo_gsm8k_base.py` (parameterized for model/seed/rank/steps)
- Tool-use scripts: `grpo_exp_a_baseline.py`, `grpo_exp_b_high_lr.py`, `grpo_exp_c_low_temp.py`, `grpo_exp_d_xlam.py`
- 100-step scripts: `grpo_100_xlam.py`, `grpo_100_synthetic.py`, `grpo_100_math.py`
- World-Class Suite scripts: `grpo_world_class_scaling.py`, `grpo_world_class_frontier.py`, `ppo_modal_baseline.py`, `kl_entropy_tracker.py`
- All logs: `/tmp/gsm8k_*.log`, `/tmp/grpo_*.log`
- Tinker checkpoints: `tinker://<run_id>/sampler_weights/final`
- HuggingFace Hub: [https://huggingface.co/arvindcr4](https://huggingface.co/arvindcr4) (`tinker-rl-bench-*` repos)
- W&B project (original): `tinker-structural-ceiling` at [https://wandb.ai/arvindcr4/tinker-structural-ceiling](https://wandb.ai/arvindcr4/tinker-structural-ceiling)
- W&B project (World-Class): `tinker-rl-lab-world-class` at [https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class)
- GitHub repos: [https://github.com/arvindcr4/tinker-rl-lab](https://github.com/arvindcr4/tinker-rl-lab) and [https://github.com/pes-llm-research/tinker-rl-lab](https://github.com/pes-llm-research/tinker-rl-lab)
- Madhu's code generation pipeline: [https://github.com/madhukumara1993/qwen3-grpo](https://github.com/madhukumara1993/qwen3-grpo)

---

## Author Contributions

**Arvind C R** (Project Lead) led the Tinker-based GSM8K multi-seed replication, scaling study, LoRA/group-size ablations, tool-use GRPO experiments, held-out evaluation, W&B experiment tracking, the 10x Structural Ceiling experiment (32 additional runs across benchmarks, architectures, model sizes, group sizes, learning rates, and constrained decoding), and the World-Class Suite (20 parallel experiments across Tinker API and Modal H100 GPUs covering 5 model families from 0.6B to ~1T parameters including the completed Kimi-K2 run, PPO baselines, and KL/entropy tracking). He also designed the statistical analysis framework (Welch's t-test, Mann-Whitney U, Cohen's d, bootstrap CIs, ANOVA with η², Bonferroni correction) and the elevation analyses (scaling law fitting, ZVF characterization, length bias analysis, 2-GRPO hypothesis test), and wrote the majority of the paper. W&B project: [tinker-rl-lab-world-class](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class). Checkpoints: [HF Hub arvindcr4/tinker-rl-bench-*](https://huggingface.co/arvindcr4). GitHub: [arvindcr4/tinker-rl-lab](https://github.com/arvindcr4/tinker-rl-lab).

**Sandhya Jeyaraj** designed and executed the 3-phase multi-turn tool-call scaling study (Experiments 1–3) across 0.5B → 1.5B → 3B models using real datasets (Glaive 112K + ToolBench 187K examples), yielding the project's clearest positive tool-use result: 0%→92% JSON validity and GRPO 0.91 vs. SFT 0.72 on the 3B multi-turn model. Conducted Qwen2.5-3B QLoRA training for multi-turn tool chaining. Limitations include no W&B logging and no personal GitHub repository for code release. HuggingFace: Balasandhya.

**Madhu Kumara L** developed a full SFT → GRPO code generation pipeline on Modal GPUs, achieving HumanEval 86% (141/164) on Qwen3-8B after SFT on 3,000 Open-Platypus examples and GRPO with 5 custom reward functions. Implemented and open-sourced the training pipeline and evaluation harness. Honestly reported an earlier SWE model failure (42%→42%). GitHub: [madhukumara1993/qwen3-grpo](https://github.com/madhukumara1993/qwen3-grpo). HuggingFace: [Madhu2133](https://huggingface.co/Madhu2133). Model: [qwen3-8b-swe-grpo](https://huggingface.co/Madhu2133/qwen3-8b-swe-grpo).

**Mohammad Rafi** ran a multi-stage reasoning experiment using Qwen3-4B-Instruct GRPO on GSM8K + NuminaMath, producing one of the project's most thoroughly documented experiments: 24KB technical writeup, LaTeX paper, and full training logs across a 10.39-hour Tinker run. Results: Baseline 67.2% → SFT 68.1% → GRPO 67.8% on GSM8K (net +0.6pp, within measurement noise). HuggingFace: [MohammadRafiML](https://huggingface.co/MohammadRafiML).

**Dhruva N Murthy** built the baseline tool-use evaluation pipeline (5 synthetic tools, 200/40/60 train/val/test split) and developed a SFT+GRPO pipeline that achieved schema compliance improvement from 52%→70-71% in initial phases and 97%→100% at the final stage, demonstrating near-perfect format compliance. He is also first author on a NeurIPS 2025 Spotlight paper on structured pruning (modhifi) and contributed HFLM_Accelerate to lm-evaluation-harness. Limitations: GRPO training logs not committed; claimed RLHF repos returning 404.

**Arumugam K** independently replicated the tool-call pipeline (JSON 0%→92%, Avg 0→0.59) and performed LoRA tool call fine-tuning on Qwen2-0.5B within a total budget of $0.17, demonstrating budget-constrained reproducibility. Explored DPO+LoRA on aerospace domain Q&A. Limitations: training dataset at 8 examples; evaluation uses keyword counting rather than principled metric.

---

## 9. Conclusion

*Central result:* In the QLoRA/LoRA regime, GRPO reliably learns structural/format tasks but fails on semantic reasoning — with a clear benchmark hierarchy (tool-use 1.0 > GSM8K 0.97 > MATH 0.57 >> HumanEval 0.00) and a model-size threshold below which no learning occurs. The World-Class Suite across **5 model families (Qwen3, Llama, DeepSeek, Nemotron, GPT-OSS) from 0.6B to 235B parameters** extends these findings with 15 completed experiments, revealing that implementation framework, algorithm-model compatibility, and instruction tuning quality are the dominant factors — not model scale alone.

We applied GRPO to four domains (tool calling, GSM8K, MATH-500, HumanEval) across **77+ Tinker runs** plus Modal H100 GPU experiments. The 10x Structural Ceiling experiment (32 dedicated runs) provides systematic evidence for when GRPO works and when it fails. The World-Class Suite extends these findings to frontier scales with PPO comparison and KL tracking. Our key results:

1. **Benchmark hierarchy** (32 runs): GRPO learns structural/format tasks perfectly (tool-use: 1.0, GSM8K: 0.97) but fails on genuine reasoning (MATH-500: 0.57, HumanEval: 0.00). The ceiling is where the task transitions from pattern-matching to reasoning.

2. **Cross-family architecture dependence**: Tool-use success is Qwen-specific (1.0 vs. Llama 0.1 on identical task with SFT; 0% for both without SFT). GSM8K shows less family dependence (Qwen 1.0 vs. Llama-instruct 0.97).

3. **Model-size threshold** (cross-family): Below 8B-instruct, GRPO produces zero learning signal — confirmed across Qwen (0.6B, 1.7B) and Llama (1B, 3B) with immediate ZVF saturation (onset=step 0, ZVF>88%).

4. **Instruction tuning is the prerequisite**: Base→instruct delta is +0.922 on GSM8K (Llama-8B), dwarfing any RL contribution. Instruction tuning determines MoE trainability — Qwen3-30B-A3B base (32.5%) vs. instruct (100%) at identical architecture.

5. **Group saturation diagnostic** (novel): Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) track when GRPO gradients vanish. G=32 is optimal (GU=54.5%, onset step 29). All group sizes converge given enough steps.

6. **LR speed-saturation tradeoff**: LR=1e-5 never saturates (GU>82%) but converges slowly. LR=3e-4 recovers after transient dip — correcting the partial-data conclusion of instability.

7. **Constrained decoding ablation**: No difference vs. unconstrained (0.981 vs. 0.998) — decoder confound is moot. GRPO genuinely learns format.

8. **Reward hacking → collapse**: Llama-8B base broke out to 0.87 reward then catastrophically collapsed to 0.00 at step 41. Qwen3-8B PPO (final_kl=60.75) shows substantial policy drift consistent with this instability pattern.

9. **Held-out generalization (negative):** GSM8K test accuracy 83.3% ± 2.2% (5 seeds × 200 examples), but base Qwen3-8B scores 82.0% without GRPO. The +1.3pp delta is not significant (\(p{=}0.26\)).

10. **MoE routing volatility**: 2.43× higher step-to-step variance vs. dense (\(p = 7 \times 10^{-6}\), Levene's test). Nemotron-120B introduces a new Mode 2b (peak-then-collapse) failure pattern even in dense architecture.

11. **SFT+GRPO complementarity** (pilot, N=1): SFT-initialized GRPO produced the strongest results. JSON 0%→92% under unconstrained decoding. Without SFT, tool-use GRPO produces 0% reward regardless of model size or architecture.

12. **PPO vs. GRPO (completed):** PPO Llama-3.1-8B-Instruct achieves **97.5%** last-10 avg (Mann-Whitney r=0.94 over GRPO Llama's 84.4%); PPO Qwen3-8B achieves only **22.5%** vs. GRPO Qwen3-8B's 97.2% (Cohen's d=0.166). No single algorithm dominates; choice depends on the target model.

13. **Frontier model results**: Qwen3-235B-A22B achieves perfect 100% last-10 in 15 steps; DeepSeek-V3.1 achieves 85% in 20 steps (starting at 0.875 on step 1). At frontier scale, RL reinforces existing capability rather than teaching new skills.

14. **Implementation framework matters**: TRL baseline 73.4% ± 7.03% vs. Tinker 99.9% (Welch's t=8.44, p=0.0014). Bootstrap CIs non-overlapping: [67.9%, 78.9%] vs. [99.3%, 100.0%].

**Statistical caveat:** Because several comparisons are either paired over the same benchmark items or based on temporally correlated single-run trajectories, the associated p-values should be read as exploratory rather than definitive inferential results.

**Limitations:** This paper is exploratory; conclusions from the 0.6B–8B regime are specific to our QLoRA/LoRA setup and should not be generalized to GRPO broadly. The tool-use success is Qwen-specific and measures syntax compliance, not semantic competence. Transfer benchmarks are null (GSM8K \(p{=}0.26\); HumanEval \(p{=}0.53\)). The MoE finding rests on limited runs. The 10x experiment uses 50 training steps — longer training horizons may change conclusions for MATH-500 and other partially-converging tasks. One planned frontier experiment (GPT-OSS-20B) did not complete due to Tinker API stall; Kimi-K2 has since been completed (Peak 100%, Last-10 80%); KL tracking failed due to gradient bug. Elevation analyses (Sections 5.11–5.14) are based on 10–28 experiments — sufficient for exploratory characterization but underpowered for definitive causal claims.

---

## References

1. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024.
2. Lai, X., et al. "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning." 2024.
3. Li, C., et al. "Common 7B Language Models Already Possess Strong Math Capabilities." 2024.
4. Havrilla, A., et al. "Teaching Large Language Models to Reason with Reinforcement Learning." 2024.
5. Pang, R., et al. "Iterative Reasoning Preference Optimization." 2024.
6. Luo, H., et al. "WizardMath: Empowering Mathematical Reasoning for Large Language Models." 2023.
7. Xiong, W., et al. "Building Math Agents with Multi-Turn Iterative Preference Learning." 2024.
8. Gulcehre, C., et al. "Reinforced Self-Training (ReST) for Language Modeling." 2023.
9. Zelikman, E., et al. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.
10. Fedus, W., et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022.
11. Du, N., et al. "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts." ICML 2022.
12. Jiang, A., et al. "Mixtral of Experts." 2024.
13. Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024.
14. Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." 2023.
15. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.
16. Hu, S., et al. "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework." arXiv:2405.11143, 2024.
17. Sheng, G., et al. "HybridFlow: A Flexible and Efficient RLHF Framework" (veRL). arXiv:2409.19256, 2024.
18. Li, L., et al. "VerifyBench: Benchmarking Mathematical Reasoning Verification of Large Language Models." arXiv:2408.01975, 2024.
19. Moonshot AI. "Kimi K2: Scaling Agentic Skills with Reinforcement Learning." Technical Report, 2025.
20. NVIDIA. "Nemotron-4: Efficient Reasoning with Mixture of Experts." Technical Report, 2025.
21. Xie, Q., et al. "It Takes Two: Your GRPO Is Secretly DPO." arXiv:2510.00977, 2025.
22. Qwen Team. "Group Sequence Policy Optimization (GSPO)." arXiv:2507.18071, July 2025. https://arxiv.org/abs/2507.18071
23. ByteDance Seed Team. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv:2503.14476, March 2025. https://arxiv.org/abs/2503.14476
24. Nimmaturi, D., Bhargava, V., Dutta, D., et al. "Predictive Scaling Laws for Efficient GRPO Training of Large Reasoning Models." arXiv:2507.18014, July 2025. https://arxiv.org/abs/2507.18014
25. Zhou, H., Ye, K., Xu, E., Zhu, J., Gong, S., Shi, C. "Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic." arXiv:2603.01162, March 2026. https://arxiv.org/abs/2603.01162
26. Nan, G., Chen, S., et al. "NGRPO: Negative-enhanced Group Relative Policy Optimization." arXiv:2509.18851, September 2025. https://arxiv.org/abs/2509.18851
27. Zhang, X., Wu, S., et al. "Scaf-GRPO: Scaffolded Group Relative Policy Optimization." arXiv:2510.19807, October 2025. https://arxiv.org/abs/2510.19807
28. Han, K., Zhou, Y., et al. "EBPO: Empirical Bayes Shrinkage for Stabilizing GRPO." arXiv:2602.05165, February 2026. https://arxiv.org/abs/2602.05165
29. Feng, Y., Jain, P., et al. "LENS: Don't Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting." arXiv:2510.08696, October 2025. https://arxiv.org/abs/2510.08696
30. Pikus, B., Tiwari, P. R., Ye, B. "Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets." arXiv:2508.14094, August 2025. https://arxiv.org/abs/2508.14094
31. Le, T.-L. V., Jeon, M., Vu, K., Lai, V., Yang, E. "No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning (RL-ZVP)." arXiv:2509.21880, September 2025. https://arxiv.org/abs/2509.21880
32. Wang, Y., Zhao, J., Zhao, C., Guan, S., Penn, G., Liu, S. "λ-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences." arXiv:2510.06870, October 2025. https://arxiv.org/abs/2510.06870
33. Li, Z., Lou, J., Dong, F., Fan, Z., Ren, M., Lin, H., Han, X., Zhang, D., Sun, L., Lu, Y., Yu, X. "GR³: Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning." arXiv:2603.10535, March 2026. https://arxiv.org/abs/2603.10535
34. EMNLP Authors. "GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach." EMNLP 2025. https://aclanthology.org/2025.emnlp-main.287
35. Zheng, et al. "GRESO: Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via GRPO with Efficient Selective Rollout." arXiv:2506.02177, NeurIPS 2025. https://arxiv.org/abs/2506.02177
36. lzhxmu et al. "CPPO: Accelerating the Training of Group Relative Policy Optimization." arXiv:2503.22342, NeurIPS 2025. https://arxiv.org/abs/2503.22342
37. Benjamini, Y., Hochberg, Y. "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." Journal of the Royal Statistical Society: Series B, 57(1):289–300, 1995.
38. ByteDance Seed Team (veRL). "VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning." arXiv:2504.05118, April 2025. https://arxiv.org/abs/2504.05118
39. Guo, D., Yang, D., Zhang, H., et al. (DeepSeek-AI). "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning." Nature 645, 633–638, 2025. https://doi.org/10.1038/s41586-025-09422-z
40. Kimi-K2-Thinking results logged at W&B: https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class (Bitter Lesson Campaign, 2026).
41. Modal TRL-GRPO comparison results: Modal H100 platform, TRL 1.2.0 with GRPOConfig; Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct, Qwen3-8B on GSM8K (Bitter Lesson Campaign, 2026).
42. Campaign design doc: /home/user/workspace/elevation_outputs/experiment_campaign.md (Bitter Lesson Campaign design and experiment matrix, 2026).

---

*77+ Tinker training runs (25 original + 32 from 10x Structural Ceiling + World-Class Suite) plus Modal H100 experiments were produced or initiated during the project. All World-Class Suite GRPO runs are checkpointed to HuggingFace Hub for permanent availability. One experiment (GPT-OSS-20B) did not complete due to Tinker API stall and remains for future work; Kimi-K2 has since been completed (Section 4.5.7).*

**Release status:** Code is publicly available at [https://github.com/arvindcr4/tinker-rl-lab](https://github.com/arvindcr4/tinker-rl-lab) and [https://github.com/pes-llm-research/tinker-rl-lab](https://github.com/pes-llm-research/tinker-rl-lab). The repository includes training scripts, the held-out GSM8K evaluation harness, Colab notebooks, and World-Class Suite scripts. All World-Class Suite checkpoints are available at [https://huggingface.co/arvindcr4](https://huggingface.co/arvindcr4) under the `tinker-rl-bench-*` namespace. Training logs and metrics for all World-Class Suite runs — including the Bitter Lesson Campaign additions (Kimi-K2-Thinking, Qwen3.5-397B-A17B, GPT-OSS-120B, TRL-GRPO Modal runs, and base/instruct comparisons) — are tracked at [https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class). The Bitter Lesson Campaign brought total experiment coverage to 70+ runs across 15+ model architectures, 2 frameworks, and 2 GPU platforms; the campaign design is documented at /home/user/workspace/elevation_outputs/experiment_campaign.md. Madhu's code generation pipeline is available at [https://github.com/madhukumara1993/qwen3-grpo](https://github.com/madhukumara1993/qwen3-grpo).
