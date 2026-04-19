# Limitations, Ethics, and Broader Impact

> NeurIPS 2026 — companion document to `paper/ethics_statement.tex`.
> Covers NeurIPS Paper Checklist Items 2 (Limitations), 7 (Code of Ethics),
> 8 (Compute Resources), 10 (Safeguards), and 11 (Broader Impacts).
> Must be kept in sync with `paper/ethics_statement.tex`; run
> `python paper_sync_audit.py` before submission.

---

## 1. Executive Summary

TinkerRL‑Bench is an **exploratory** case study of critic‑free reinforcement
learning (GRPO) for post‑training small and mid‑sized language models. It
contributes (i) a multi‑platform benchmark harness that runs identical
GRPO/PPO/DPO workloads on Tinker, Modal, TRL, and Google Colab; (ii) a set
of diagnostic metrics (Zero‑Variance Fraction, reward stability, length
bias); and (iii) the first systematic audit of a closed‑source RL‑training
API against open‑source baselines on GSM8K and xLAM function calling.

Because the work exists at the boundary between a commercial SaaS (Tinker)
and the open‑source RL ecosystem, a large share of this document is
devoted to describing exactly what we can and cannot claim from each
platform, and to itemising the compute, cost, and environmental footprint
so that reviewers and replicators can make an informed judgement.

---

## 2. Dual‑Use Analysis: Misuse of Math‑Reasoning and Tool‑Use RL

### 2.1 Threat Model

The artefacts we release are:

1. Training scripts that apply GRPO to GSM8K grade‑school math and to the
   Salesforce xLAM‑Function‑Calling‑60k schema.
2. LoRA adapters (rank 8–64) on the HuggingFace Hub under
   `arvindcr4/tinker-rl-bench-*`.
3. Diagnostic code (Zero‑Variance Fraction, reward‑stability, length‑bias,
   group‑saturation).
4. Raw CSV reward logs for every run, including failed runs.

### 2.2 Pathways Considered

**M1. Weaponisation of mathematical reasoning.** GRPO could be re‑pointed
from GSM8K to chemistry olympiad problems, cryptographic key recovery, or
financial fraud. We judge the *marginal* uplift from our release to be
small because:

- Base‑model control for Qwen3‑8B on GSM8K already scores 82.0% held‑out.
  Full GRPO adds **only +1.3 points** (83.3%; *p*=0.26, not significant).
  The model's capability is overwhelmingly pre‑existing.
- Public implementations of GRPO already exist (DeepSeekMath, OpenRLHF,
  veRL, TRL) and are at least as capable as our pipeline.
- GSM8K's reward is exact‑match against human gold answers — it does not
  transfer to the misuse domains above without a new labelled reward
  signal, which itself requires expertise and data that we do not provide.

**M2. Reward hacking / specification gaming.** GRPO optimises a proxy
reward; like all policy‑gradient RL it is vulnerable to specification
gaming (Skalse et al. 2022; Krakovna et al. 2020). Our own length‑bias
analysis flags **4 of 11** GRPO runs as exhibiting the verbosity trap
(peak before 65% of training, terminal reward <90% of peak). Users who
apply our scripts to under‑specified reward models in production may
observe models that look better on held‑out metrics while silently
optimising against unintended proxies. We release the diagnostics needed
to detect this (Section 4.4 of the main paper).

**M3. Circumventing safety training via the tool‑use pipeline.** The
tool‑use track teaches a base model to emit schema‑valid JSON function
calls on a five‑tool synthetic schema. An adversary could in principle
swap the schema to teach the model to emit jailbreak prompts or invoke an
attacker‑controlled tool. Mitigations:

- We release **adapters only**, not merged weights. The
  safety‑trained instruct checkpoint must be applied separately so that
  instruct‑layer guardrails stay active.
- Model cards state explicitly that tool‑use adapters are trained on a
  narrow synthetic schema and have **not** been safety‑evaluated for
  open‑ended function calling.
- We do **not** ship an execution harness that would actually dispatch
  emitted calls.

**M4. Compute‑efficiency externalities.** Any improvement in training
efficiency lowers the cost of adversarial fine‑tuning at scale. The group
of work around G=32, LoRA, and short horizons is already widely known;
our marginal contribution here is modest. We flag the externality
explicitly so reviewers can weigh it.

### 2.3 Residual Risk Assessment

Weighing M1–M4 against the reproducibility, calibration, and pedagogical
benefits of releasing TinkerRL‑Bench, we judge the residual risk to be
low and acceptable for publication under the standard mitigations: adapters
not merged weights, documented intended use, and diagnostic tooling
included alongside the training code.

### 2.4 Responsible‑Use Guidelines (Mandatory in Model Cards)

1. Do **not** use the released adapters or methods to intentionally
   bypass safety training of aligned models.
2. Do **not** use distillation methods or SFT→GRPO pipelines to transfer
   harmful capabilities between checkpoints.
3. Cite and credit the base‑model creators (Alibaba for Qwen, Meta for
   Llama, DeepSeek, NVIDIA for Nemotron, Moonshot for Kimi, OpenAI for
   GPT‑OSS) and respect their licenses.
4. Do **not** use the tool‑use adapters in any execution environment that
   dispatches real‑world side effects (filesystem writes, HTTP, payments)
   without an independent safety layer.
5. Report any discovered vulnerabilities, jailbreaks, or misuse potential
   to the maintainers via GitHub Issues on
   [pes-llm-research/tinker-rl-lab](https://github.com/pes-llm-research/tinker-rl-lab)
   and the mirror
   [arvindcr4/tinker-rl-lab](https://github.com/arvindcr4/tinker-rl-lab).

---

## 3. Compute Cost Accounting

All figures below are actual out‑of‑pocket spend for the authors. In‑kind
compute (PES University A100 node) is priced at list GCP rates for
transparency but was not paid out of pocket. Dollar figures are USD.

| Platform | Use | Runs | Cost (USD) |
|---|---|---:|---:|
| Tinker SDK v0.16.1 | GRPO on 4B/8B (original suite) | 25 | \$40–55 |
| Tinker SDK v0.16.1 | 10x Structural Ceiling ablation | 32 | \$65 |
| Tinker SDK v0.16.1 | World‑Class Suite (frontier / MoE) | 13 | \$25–30 |
| Modal H100 | PPO baselines (Qwen3‑8B, Llama‑3.1‑8B) | 2 | \$12–18 |
| Modal H100 | KL‑tracking run (failed gradient bug) | 1 | \$2–4 |
| Google Colab Pro (T4) | 0.5B–3B QLoRA SFT+GRPO | — | \$10/person |
| NVIDIA L4 (TRL baseline) | Qwen2.5‑0.5B GSM8K, 5 seeds × 125 steps | 5 | \$3–5 |
| HuggingFace Hub | Model + adapter hosting | — | \$0 |
| Weights & Biases | Experiment tracking (academic tier) | — | \$0 |
| **Total authors' out‑of‑pocket** | | | **\$130–140** |

**In‑kind compute (valued, not paid):**

- PES University LLM Research Group A100 node — ~60 GPU‑hours, equivalent
  to ~\$270 at GCP a2‑highgpu list price.

**Failed‑run accounting:** Approximately \$30–35 of Tinker spend was
consumed by runs that ultimately failed (JWT expiry, GPT‑OSS‑20B API
stall, Kimi‑K2 initial stall before successful re‑run, KL‑tracking
gradient bug). We disclose this because hiding it would understate the
true resource cost of reproducing the work by ~25%.

---

## 4. Carbon Footprint

Methodology follows Patterson et al. (2024) and Strubell et al. (2019):
energy = GPU‑hours × TDP × PUE; CO₂ = energy × grid intensity.

### 4.1 Assumed Constants

| Quantity | Value | Source |
|---|---|---|
| NVIDIA H100 SXM5 TDP | 700 W | NVIDIA datasheet |
| NVIDIA A100 80GB TDP | 400 W | NVIDIA datasheet |
| NVIDIA L4 TDP | 72 W | NVIDIA datasheet |
| NVIDIA T4 TDP | 70 W | NVIDIA datasheet |
| Cloud data‑centre PUE | 1.1 | Google / AWS 2023 sustainability reports |
| US average grid intensity | 0.367 kg CO₂e/kWh | EPA eGRID 2023 |
| Indian average grid intensity | 0.716 kg CO₂e/kWh | CEA 2023 CO₂ baseline database |

### 4.2 Per‑Platform Estimate

| Platform | GPU‑h | TDP (W) | PUE | Energy (kWh) | CO₂ (kg) |
|---|---:|---:|---:|---:|---:|
| Tinker (assumed 1× H100) | ~950 | 700 | 1.1 | ~732 | ~269 |
| Modal H100 (US‑East) | ~18 | 700 | 1.1 | ~14 | ~5.1 |
| PES A100 (in‑kind) | ~60 | 400 | 1.1 | ~26 | ~19 |
| Colab Pro T4 (asia‑south1) | ~40 | 70 | 1.2 | ~3.4 | ~2.4 |
| NVIDIA L4 (TRL baseline) | ~10 | 72 | 1.1 | ~0.8 | ~0.3 |
| **Project total (central)** | | | | **~776** | **~296** |

### 4.3 Interpretation and Uncertainty

Central estimate: **~296 kg CO₂‑eq** across the entire project, comparable
to a single round‑trip Bengaluru→Delhi economy flight (~300 kg).

The **dominant uncertainty** is the Tinker GPU‑hour count: Tinker does not
expose hardware telemetry, so GPU type, TDP, and utilisation are inferred.
Sensitivity analysis:

- **Pessimistic** (2× GPU‑hours, H100 at 100% TDP): Tinker contribution
  could reach ~540 kg CO₂‑eq.
- **Optimistic** (more efficient accelerators than H100, 50% average
  utilisation): Tinker contribution could be as low as ~130 kg CO₂‑eq.

All numbers should be treated as **order‑of‑magnitude** upper bounds on
the cost of reproducing our work; replicators can skip the failed runs
and the ablation sweep.

### 4.4 Mitigations

We did **not** purchase carbon offsets. Instead:

- All trained checkpoints and adapters are published on HuggingFace Hub so
  downstream users do not need to re‑train.
- Step‑level CSV logs are shipped so learning curves can be inspected
  without re‑running.
- Failed runs are documented explicitly so replicators can skip them.

Reproducibility‑by‑artefact is, in our view, a more durable mitigation
than offsets.

---

## 5. Data Provenance

No private data, PII, or licensed proprietary content is used. No human
annotators were employed. No IRB review was required.

| Dataset | Scale | Licence | Used For | Citation |
|---|---|---|---|---|
| **GSM8K** | 7,473 train / 1,319 test | MIT | Math reasoning GRPO + held‑out eval | Cobbe et al. 2021, [github.com/openai/grade-school-math](https://github.com/openai/grade-school-math) |
| **Salesforce xLAM‑Function‑Calling‑60k** | 60,000 examples | CC BY 4.0 | Tool‑use GRPO (real‑data track) | Liu et al. 2024, [HuggingFace](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| **HumanEval** | 164 problems | MIT | Code generation pass@k | Chen et al. 2021 |
| **NuminaMath** | ~860k pairs | Apache 2.0 | Multi‑stage SFT (Rafi's pipeline) | Li et al. 2024 |
| **Open‑Platypus** | 25k total, 3k used | CC BY‑NC 4.0 | Code‑gen SFT warm‑up (Madhu's pipeline) | Lee et al. 2023 |
| **Synthetic 5‑tool corpus** | authored by the authors | MIT | Tool‑use saturation analysis | This work |

### 5.1 Attribution and Licence Compliance

- **Open‑Platypus CC BY‑NC 4.0** restricts commercial use. Our release is
  academic‑use only; the adapter trained on Open‑Platypus is flagged
  non‑commercial in its model card.
- **xLAM CC BY 4.0** requires attribution to Salesforce Research. The
  Acknowledgments section of the main paper and the model cards for
  xLAM‑derived adapters both include the required attribution.
- All other datasets above are MIT or Apache 2.0 and are redistributed or
  reused within those licenses' terms.

### 5.2 Known Limitations of the Chosen Datasets

- **GSM8K cultural and linguistic bias.** US‑centric names, currency,
  and sports references. A ~2% human‑labelling error rate is documented
  in GSM1k (Zhang et al. 2024).
- **xLAM synthetic schema skew.** Schemas are cleanly typed; ambiguous
  arguments, error‑handling, and multi‑turn tool calls are
  under‑represented.
- **HumanEval contamination.** Documented contamination concerns against
  frontier pretraining corpora (Riddell et al. 2024) imply that reported
  pass@k may be inflated for any model trained on post‑2021 web snapshots.
  We report HumanEval as an exploratory track only.

### 5.3 What We Did **Not** Use

- No web scraping.
- No user data from any service (ours or third‑party).
- No copyrighted text, code, or media beyond what is contained in the
  datasets above under their permissive licenses.
- No human participants (no consent forms, no surveys, no preference
  annotation tasks).

---

## 6. Closed‑Source Tinker Acknowledgment

### 6.1 What Tinker Is and Is Not

Tinker (Thinking Machines Inc., [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker))
is a managed LLM fine‑tuning and inference API. The Python SDK
`tinker==0.16.1` we use exposes:

- Custom loss via `forward_backward_custom(loss_type_input="logprobs")`.
- A limited optimiser surface (Adam with tuneable β and ε).
- Standard LoRA hyperparameters (rank, α, dropout, target modules).

It does **not** expose:

- The exact server‑side GRPO loss formulation.
- The reward normalisation or baseline‑subtraction scheme.
- Minibatch construction or gradient‑accumulation strategy.
- Hardware configuration (GPU type, interconnect, topology).
- System‑level telemetry (energy, throughput, queue time).

### 6.2 Consequences for Our Claims

Tinker results measure the **platform's** implementation of GRPO, not an
abstract algorithmic specification. A reader asking "why does Tinker GRPO
score 99.9% on GSM8K while open‑source TRL GRPO scores 73.4% on the same
task" cannot fully answer that from our data — we can rule out a handful
of candidate explanations (seed variance, model‑size confound via the
same‑model Qwen3‑8B comparison) but **not the implementation itself**.

We therefore:

- Draw **quantitative** conclusions only from the open‑source side (TRL,
  veRL on Modal H100, OpenRLHF) where every hyperparameter is auditable.
- Use Tinker results as an **upper bound** on what a carefully engineered
  production stack can achieve for critic‑free RL at our scales.
- Mark every Tinker‑only figure and summary statistic with "†" and an
  accompanying footnote.

### 6.3 Reproducibility Commitments

1. All Tinker experiment scripts, configs, step‑level JSON logs, and
   per‑run W&B projects are archived in both repositories.
2. Figures derived solely from Tinker data are marked "†" with a footnote
   that independent replication requires Tinker API access.
3. Primary significance tests (Welch's t, ANOVA) are computed on
   open‑source data; Tinker is reported descriptively.
4. If an equivalent open‑source service becomes available or if Tinker
   open‑sources its implementation, we commit to re‑running the Tinker
   experiments on the open backend and issuing a revised version of this
   paper.

### 6.4 API Key Hygiene

- The Tinker API key (prefix `tml-...`) is stored only in the authors'
  password manager.
- `.env.example` in the repo contains **only** placeholder text; no real
  key has been committed to git history in either repo.
- The key is rotated whenever a failure mode suggests possible leakage.
  A rotation log is maintained internally (not publicly committed) so
  that incidents can be audited post‑hoc.

---

## 7. Known Methodological Limits

These limits are enumerated here in addition to the infrastructure
failures documented in `paper/main.tex` § Limitations (JWT expiry,
Modal timeouts, KL‑tracking bug, W&B step‑level data loss).

### 7.1 Statistical Limits

- **Short training horizons (30–50 steps).** Tinker runs are early‑training
  snapshots. Long‑horizon effects — reward hacking, catastrophic
  forgetting, late‑stage policy collapse — are unlikely to manifest at
  this scale. Asymptotic claims are unsupported.
- **Single‑seed Tinker experiments.** Each Tinker configuration ran once
  due to cost. No variance estimates, no significance tests on Tinker
  data. Henderson et al. (2018) recommends ≥5 seeds; we hit this bar only
  for the TRL baseline and the held‑out GSM8K evaluation.
- **5 seeds on held‑out GSM8K.** Patterson et al. (2024) recommend 10+
  seeds for reliable RL comparisons. Compute constraints forced us to 5.
- **Bootstrap CI assumes i.i.d.** Correlations from shared initialisation
  or data ordering could narrow true confidence intervals.
- **30B MoE scaling analysis uses 3 seeds** due to compute constraints.
- **Un‑corrected p‑values for secondary analyses.** Bonferroni‑surviving
  claims are explicitly flagged in the main paper; other tests are
  descriptive.

### 7.2 Methodological Confounds

- **Train‑set reward as primary Tinker metric.** Only GSM8K was followed
  up with held‑out evaluation. Tool‑use and xLAM remain train‑set‑only; we
  cannot separate memorisation from generalisation on those tracks.
- **LoRA only, no full fine‑tuning.** Rank 8–64 LoRA everywhere. We have
  not tested whether full FT would re‑order the library/algorithm
  comparisons.
- **Cross‑platform hardware confounding.** Tinker, Modal, Colab, and PES
  A100 use different accelerators, memory hierarchies, and training
  stacks. Observed differences in reward dynamics are not purely
  algorithmic.
- **Hyperparameter sensitivity not exhaustively swept.** Defaults from
  Tinker cookbook + TRL docs; full LR × rank × batch × group‑size grid
  would strengthen the empirical contribution.
- **Exploratory, not confirmatory.** No pre‑registration of primary
  hypotheses.

### 7.3 Scope Limits

- **Model scale.** 0.6B–235B only. 70B dense and 1T MoE extrapolations
  (beyond Kimi‑K2) unreliable.
- **Benchmark coverage narrow.** GSM8K, MATH‑500 (exploratory), HumanEval
  (subset), synthetic/xLAM tool‑use. No MT‑Bench, ArenaHard, HarmBench,
  ToxicChat, or TruthfulQA.
- **No human preference data.** Verifiable rewards only (exact‑match,
  unit‑test, schema‑validity). Findings should not be extrapolated to
  reward‑model‑based RLHF without further work.
- **Geographic and demographic scope.** All authors based at one
  institution in Bengaluru, India. Benchmark choices, prompt phrasings,
  and evaluation design reflect that context. Non‑English tasks and
  non‑Qwen / non‑Llama architectures are out of scope.

### 7.4 Carbon Accounting Limits

See Section 4.3. Tinker GPU‑hour and TDP are inferred; grid intensity is
region‑average, not marginal; reported CO₂ is order‑of‑magnitude.

---

## 8. Positive Impacts

- **Democratising RL post‑training research.** Total out‑of‑pocket cost of
  ~\$130 demonstrates that rigorous RL post‑training benchmarking is
  feasible for under‑resourced groups. The repository ships Docker
  images, cost‑annotated run logs, and step‑by‑step `REPRODUCE.md`
  instructions.
- **Improving reproducibility standards.** The multi‑seed protocol, the
  abstract‑scope audit script (`abstract_scope_audit.py`), and the
  anonymisation audit (`anonymization_repro_audit.py`) directly address
  the RL reproducibility crisis documented by Henderson et al. (2018) and
  Pineau et al. (2020).
- **Fair cross‑library comparison.** Standardised reward functions and
  hyperparameter mappings reduce confounding variables between TRL,
  Tinker, veRL, OpenRLHF, and SkyRL.
- **Platform‑dependent variance as a first‑class variable.** By running
  identical configurations on a closed API (Tinker) and an open cloud
  (Modal H100 + veRL), we surface implementation‑framework effects
  quantitatively (TRL 73.4% vs. Tinker 99.9% on the same task,
  *p*=0.0014).

---

## 9. Mitigations and Release Hygiene

- All released model checkpoints are LoRA adapters over already‑released
  base models (Qwen, Llama, DeepSeek, Nemotron, Kimi, GPT‑OSS) and do
  **not** introduce new safety risks beyond those already present in the
  base models.
- Each adapter has a **model card** documenting intended use, training
  data, training hardware, known limitations, and licence.
- The benchmark focuses on **verifiable** reward domains (math,
  schema‑valid JSON, unit tests), which have lower misuse potential than
  open‑ended generation rewards.
- No jailbroken, safety‑weakened, or undisclosed checkpoints are released.
- No execution harness for tool‑use is shipped.
- Reviewers are encouraged to apply the statistical and diagnostic
  methods from this work to safety‑critical domains (alignment,
  harmlessness, helpfulness) in future work.

---

## 10. Sync With Paper

This document must stay in sync with `paper/ethics_statement.tex`. The
matching Tables are:

- `paper/ethics_statement.tex` Table `tab:compute_spend` ↔ Section 3 here.
- `paper/ethics_statement.tex` Table `tab:carbon2` ↔ Section 4 here.
- `paper/ethics_statement.tex` § Data Provenance ↔ Section 5 here.
- `paper/ethics_statement.tex` § Closed‑Source Tinker Acknowledgment ↔
  Section 6 here.
- `paper/ethics_statement.tex` § Known Methodological Limits ↔ Section 7
  here.

Run `python paper_sync_audit.py` before every submission revision.

---

## 11. Contact and Disclosure Channel

- GitHub Issues (public): https://github.com/pes-llm-research/tinker-rl-lab/issues
- Mirror: https://github.com/arvindcr4/tinker-rl-lab/issues
- Responsible disclosure (security / jailbreak / misuse concerns):
  open a private security advisory on either repository, or email the
  corresponding author listed on the paper PDF.
