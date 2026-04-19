# NeurIPS 2026 Paper Checklist — Final Pass

> Final, honest pass of the NeurIPS 2026 Paper Checklist for *TinkerRL-Bench*
> (`paper/main.tex`). Each item records the answer (`Yes` / `Partial` / `NA`),
> the location in the PDF of `paper/main.pdf` (37 pages, built 2026-04-19), a
> justification grounded in the paper text, and — where applicable — a flag
> noting where the previously-filled answer (in `NEURIPS_CHECKLIST.md` and the
> `\section*{NeurIPS Paper Checklist}` block in `paper/main.tex`) was *not*
> supported by what the paper actually says. Flags are prefixed `[AUDIT]`.
>
> Anchor map (from `pdftotext` of the current build of `paper/main.pdf`):
>
> | Section | Label | Page |
> | --- | --- | --- |
> | §1 Introduction | `sec:intro` | 2 |
> | §2 Related Work | `sec:related` | 4 |
> | §3 Benchmark Design | `sec:benchmark` | 7 |
> | §3.3 Hyperparameter Mapping (Table 3 on p. 8) | `tab:hyperparams` | 7–8 |
> | §4 Experimental Setup | `sec:setup` | 8 |
> | §4.3 Statistical Methodology | — | 9 |
> | §4.4 Compute Resources | — | 10 |
> | §5 Results | `sec:results` | 11 |
> | §6 Reproducibility | `sec:reproducibility` | 24 |
> | §7 Limitations | `sec:limitations` | 24 |
> | §7.1 Infrastructure Failures | `sec:infra_failures` | 24 |
> | §7.2 Methodological Limitations | `sec:method_limits` | 25 |
> | §7.3 Length Bias / Reward Instability | `sec:length_bias` | 26 |
> | §7.4 Broader Impact | `sec:impact` | 26 |
> | §8 Discussion | `sec:discussion` | 27 |
> | §9 Conclusion | `sec:conclusion` | 29 |
> | Appendix A — Compute Resources | `app:compute` | 33 |
> | Appendix B — Hyperparameter Tables | `app:hyperparams` | 33 |
> | Appendix C — Additional Results | `app:results` | 34 |
> | NeurIPS Checklist (in-paper) | — | 34–37 |

---

## 1. Claims
- **Answer:** Yes
- **Location:** Abstract (p. 1), §1 Introduction (p. 2), §5 Results (pp. 11–23)
- **Justification:** The abstract on p. 1 enumerates the five empirical
  contributions — (i) a 73-percentage-point implementation gap across 7
  libraries, (ii) model-dependent GRPO/PPO preference, (iii) frontier
  collapse on Nemotron-120B, (iv) zero-variance fraction (ZVF) as a leading
  diagnostic of GRPO failure, and (v) policy-drift proxies for
  KL-free monitoring. Each claim is supported in §5:
  implementation gap in §5.2 (p. 11), algorithm-by-model interaction in
  §5.12 (p. 17), frontier instability in §5.11 (p. 17), ZVF analysis in
  §5.14 (p. 20), and reward-trajectory proxies in §7.3 (p. 26). Scope
  (LoRA-only, three task families, 0.6B–235B evaluated, 0.6B–671B
  measured) is stated in §4.1 (p. 8) and §1 (p. 2).
- **[AUDIT]** The existing in-paper checklist answer (p. 35) says
  "Section 5 presents empirical evidence for each claim across **15
  completed experiments**." This number is inconsistent with the abstract
  (44 experiments), §4.4 (14 Tinker + 6 Modal = 20 core experiments, later
  expanded to 70+ in §5.17, p. 23), and §5.14 which reports "28
  experiments with step-level ZVF traces". Recommend rewording to "across
  the 44 controlled experiments reported in §5 (plus the 32-run Bitter
  Lesson extension in §5.17)". This is fixed in the `checklist.tex` file
  produced alongside this document.

## 2. Limitations
- **Answer:** Yes
- **Location:** §7 Limitations (p. 24), §7.1 Infrastructure Failures
  (p. 24), §7.2 Methodological Limitations (p. 25), §7.3 Length Bias and
  Reward Instability Analysis (p. 26), §8.4 Limitations of Proxy Metrics
  (p. 29)
- **Justification:** §7.1 (p. 24) documents that 11/14 Tinker experiments
  were interrupted by JWT token expiry and 4/6 Modal experiments timed
  out, naming the specific runs. §7.2 (p. 25) discusses LoRA-only scope,
  single-seed Tinker runs, train-set reward metric, and 0.6B–235B scale
  ceiling. §7.3 (p. 26) treats length bias / reward instability. §8.4
  (p. 29) discusses the substitution of reward-trajectory proxies for
  direct KL measurement. These match the four limitation categories
  enumerated in the in-paper checklist answer on p. 35.

## 3. Theory, Assumptions and Proofs
- **Answer:** NA
- **Location:** — (empirical paper)
- **Justification:** The paper presents no formal theorems. Objective
  definitions for GRPO and PPO appear as background in §3 (p. 7) but are
  not claimed as original theoretical results. `\answerNA` is therefore
  correct.

## 4. Experimental Result Reproducibility
- **Answer:** Partial
- **Location:** §6 Reproducibility (p. 24), §4.2 Training Protocol (p. 8),
  §7.1 Infrastructure Failures (p. 24), Appendix A Compute Resources
  (p. 33), `REPRODUCE.md`, `Dockerfile`, `requirements.txt`,
  `utils/seed.py`
- **Justification:** Reproducibility is genuinely partial by construction.
  (a) Modal experiments are fully reproducible: §6 (p. 24) enumerates
  Docker, pinned dependencies, centralized seed management
  (`utils/seed.py`), and 5-seed runs (seeds $\{42, 123, 456, 789, 1024\}$
  per §4.2 p. 8 and §5.1 p. 11). `REPRODUCE.md` gives step-by-step
  commands. (b) Tinker API runs are *not* independently reproducible
  because GPU type, driver, and scheduler are proprietary to the Tinker
  platform (§7.1, p. 24; §4.4, p. 10). Tinker-derived numbers in figures
  and tables are marked with a `†` dagger per the legends on p. 12. This
  is the strongest defensible answer; `Yes` would overstate.

## 5. Open Access to Data and Code
- **Answer:** Yes
- **Location:** §6 (p. 24), Abstract links (p. 1), `README.md`,
  `REPRODUCE.md`, `ARTIFACT.md`
- **Justification:** Code is public at
  [`github.com/arvindcr4/tinker-rl-lab`](https://github.com/arvindcr4/tinker-rl-lab);
  mirror at `pes-llm-research/tinker-rl-lab`. Model checkpoints are on
  HuggingFace under `huggingface.co/arvindcr4/tinker-rl-bench-*` (model
  card template in `huggingface/MODEL_CARD_TEMPLATE.md`). All W&B runs
  live in the public project
  [`arvindcr4-pes-university/tinker-rl-lab-world-class`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class)
  (referenced from §6, p. 24). Training datasets (GSM8K, HumanEval,
  NoRobots, synthetic tool-use) are public and cited in Table 1 (p. 7)
  and §3.1 (p. 7).
- **[AUDIT]** The in-paper checklist answer on p. 35 states the code is
  released "under MIT licence", but the actual `LICENSE` file at the
  repository root is **Apache License 2.0**. The fix in `checklist.tex`
  states the correct licence (Apache-2.0). Item 12 (Licenses) has the
  same error and is corrected in parallel.

## 6. Experimental Setting/Details
- **Answer:** Yes
- **Location:** §4 Experimental Setup (p. 8), §4.1 Models (p. 8),
  §4.2 Training Protocol (p. 8), Table 3 Hyperparameter mapping (p. 8),
  Appendix B Hyperparameter Tables incl. Table 18 sweep ranges (p. 33–34),
  `atropos/configs/*.yaml`
- **Justification:** §4.2 (p. 8) states: LoRA rank 32, learning rate
  $10^{-4}$, Adam ($\beta_1{=}0.9$, $\beta_2{=}0.95$, $\epsilon{=}10^{-8}$),
  5 seeds for Modal runs. Table 3 (p. 8) gives the cross-library
  hyperparameter mapping. Appendix B (p. 33, Table 18) enumerates sweep
  ranges for learning rate, clip range, entropy coefficient, $\gamma$, and
  GAE $\lambda$. Per-task YAML configs are version-controlled in
  `atropos/configs/`. Data splits follow the standard GSM8K train/test
  partitions (§3.1, p. 7). `Yes` is defensible.

## 7. Experiment Statistical Significance
- **Answer:** Partial
- **Location:** §4.3 Statistical Methodology (p. 9–10), Table 6
  power-analysis summary (p. 9), Table 7 BH-adjusted $p$-values (p. 10),
  §5.1 Variance decomposition (p. 12), Appendix C Figure 18 5-seed violin
  (p. 34)
- **Justification:** For Modal/TRL multi-seed arms we report mean ± SE
  with 95 % bootstrap CIs via `rliable` (§4.3, p. 9; §5.1 TRL baseline,
  p. 12: $\bar{x}=0.734$, $\sigma=0.0703$, IQM $=0.747$, bootstrap CI
  $[0.6720, 0.7820]$). Welch and Mann–Whitney tests with
  Benjamini–Hochberg correction are reported in Table 7 (p. 10). Tinker
  API runs are single-seed (§4.3, p. 9) and are explicitly flagged as
  descriptive-only; no statistical significance is claimed for
  Tinker-only comparisons. Hence `Partial`, following Colas et al. (2019)
  and Jordan et al. (2024). `Yes` would be dishonest given Tinker
  single-seed coverage.

## 8. Experiments Compute Resources
- **Answer:** Partial
- **Location:** §4.4 Compute Resources (p. 10), Appendix A (p. 33),
  `COMPUTE.md`
- **Justification:** §4.4 (p. 10) and Appendix A (p. 33) split compute
  into 14 Tinker API runs, 6 Modal H100 SXM5 (80 GB) runs, and TRL
  baselines on NVIDIA L4 (24 GB); total $\sim$1,200 A100-equivalent
  GPU-hours. `COMPUTE.md` gives per-experiment wall-clock and estimated
  cost. For Tinker runs the exact GPU type is undisclosed by the
  platform, so GPU-hours are inferred from billing — hence `Partial`.

## 9. Code of Ethics
- **Answer:** Yes
- **Location:** §7.4 Broader Impact (p. 26), `ethics_statement.tex`
- **Justification:** No human subjects, no private or sensitive data, no
  proprietary training corpora. All training datasets are public research
  datasets (GSM8K, HumanEval, NoRobots, synthetic tool-use). Risks are
  discussed in §7.4 (p. 26) and `ethics_statement.tex`.

## 10. Broader Impacts
- **Answer:** Yes
- **Location:** §7.4 Broader Impact (p. 26), §8.4 (p. 29),
  `LIMITATIONS_AND_IMPACT.md`, `ethics_statement.tex`
- **Justification:** §7.4 (p. 26) discusses positives (lower barrier to
  RL post-training research, reproducibility auditing, surfacing
  platform-dependent variance), risks (reward hacking, alignment concerns
  under proxy rewards, compute-access disparities, potential misuse of
  fine-tuned models), and a carbon-footprint estimate (0.4–1.2 kg
  CO$_2$-eq for Modal H100 runs; Tinker footprint unestimable).

## 11. Safeguards
- **Answer:** NA
- **Location:** §7.4 (p. 26), `huggingface/MODEL_CARD_TEMPLATE.md`
- **Justification:** The released artifacts are LoRA adapters fine-tuned
  from already-public base models (Qwen, Llama, Nemotron) on GSM8K,
  HumanEval, and synthetic tool-use data. No new high-risk capabilities
  are introduced beyond those of the base models (§7.4, p. 26). Model
  cards are emitted via `MODEL_CARD_TEMPLATE.md`. `NA` is therefore
  appropriate; `Yes` would over-claim, and `No` would be incorrect.

## 12. Licenses
- **Answer:** Yes
- **Location:** §6 (p. 24), Checklist item 12 in-paper (p. 37),
  `LICENSE`, `huggingface/MODEL_CARD_TEMPLATE.md`
- **Justification:** Credited assets and their licences:
  - GSM8K [Cobbe et al. 2021] — MIT licence
  - HumanEval (OpenAI) — MIT licence
  - NoRobots (HuggingFaceH4) — CC-BY-NC-4.0 (chat SFT; §3.1 / Table 1, p. 7)
  - Synthetic tool-use data — self-generated from public API docs, no
    licence restrictions
  - Qwen model family — Apache 2.0
  - Llama model family — Meta Llama Community Licence
  - TRL, PEFT, Transformers — Apache 2.0 (HuggingFace)
  - Modal — commercial platform; terms of service respected
  - Tinker API — proprietary; used under standard API terms
  - **Our code: Apache-2.0** (per `LICENSE` at repo root)
- **[AUDIT]** The in-paper checklist answer on p. 35 says "MIT licence"
  for our code — this is wrong. The repository `LICENSE` file is
  Apache-2.0. Corrected in `checklist.tex` and in item 5 above.

## 13. Assets
- **Answer:** Yes
- **Location:** §6 (p. 24), `README.md`, `REPRODUCE.md`, `ARTIFACT.md`,
  `COMPUTE.md`, `BASELINES.md`, `BENCHMARKS_COMPARISON.md`,
  `huggingface/MODEL_CARD_TEMPLATE.md`
- **Justification:** New assets released:
  1. The *TinkerRL-Bench* benchmark suite (harness, evaluation scripts,
     analysis notebooks) at `github.com/arvindcr4/tinker-rl-lab` under
     Apache-2.0, documented in `README.md`, `REPRODUCE.md`, `ARTIFACT.md`,
     `COMPUTE.md`, `BASELINES.md`, `BENCHMARKS_COMPARISON.md`,
     `LIMITATIONS_AND_IMPACT.md`.
  2. LoRA adapter checkpoints from the successful Modal experiments on
     HuggingFace Hub (`huggingface.co/arvindcr4/tinker-rl-bench-*`), each
     emitted via the `MODEL_CARD_TEMPLATE.md` documenting intended use,
     training data, and limitations.
  3. All experiment logs on Weights & Biases
     (`arvindcr4-pes-university/tinker-rl-lab-world-class`).
  Checkpoints from failed (JWT-interrupted) Tinker runs are intentionally
  *not* uploaded (§6, p. 24; §7.1, p. 24).
- **[AUDIT]** Dataset datasheet is referenced in the original checklist
  but no `DATASHEET.md` currently exists in the repository — the synthetic
  tool-use data is self-generated, so a datasheet is appropriate. This is
  a follow-up TODO for Task 10+; for the current submission the model
  cards + `ARTIFACT.md` cover the released artefacts.

## 14. Crowdsourcing and Research with Human Subjects
- **Answer:** NA
- **Location:** — (no human subjects)
- **Justification:** No crowdsourcing or human subjects research was
  performed.

## 15. IRB Approvals
- **Answer:** NA
- **Location:** — (no human subjects)
- **Justification:** No human subjects research — IRB approval does not
  apply.

## 16. Declaration of LLM Usage
- **Answer:** Yes
- **Location:** §3 Benchmark Design / §4 Experimental Setup (pp. 7–8),
  Acknowledgments (p. 30)
- **Justification:** LLMs (Qwen-3, Qwen-2.5-Coder, Qwen-3.5, Llama-3.1,
  Nemotron, GPT-OSS, Kimi-K2) are the *subjects* of evaluation — their
  role as RL fine-tuning targets is documented in §3 (p. 7) and §4.1
  (p. 8). For paper preparation, GitHub Copilot was used for boilerplate
  experiment scripts and LaTeX bibliography formatting; this is the only
  LLM-in-the-loop component of the authoring workflow.
- **[AUDIT]** The in-paper checklist claim (p. 37) "LLMs were **not** used
  to draft or revise this paper" is stronger than the repository evidence
  supports: `autoresearch.sh`, `reports/final/chatgpt_pro_*`, and
  `chatgpt_session_a_q1_5.txt` show that ChatGPT Pro was used for
  reviewer-style feedback on drafts (feedback was incorporated as
  revisions). The honest phrasing is: *LLMs were used as review and
  editing aids (ChatGPT Pro for reviewer-style critique; GitHub Copilot
  for code/bibliography scaffolding); all scientific claims, experimental
  design, results, and analysis are human-authored and verified against
  the logged W\&B traces.* This is the phrasing used in `checklist.tex`.

---

## Summary of audit flags (to fix in `paper/sections/checklist.tex`)

| # | Item | Issue | Fix applied in `checklist.tex` |
| --- | --- | --- | --- |
| 1 | Claims | "15 completed experiments" contradicts 44-experiment corpus | Reworded to "across the 44 controlled experiments in §5 and the 32-run Bitter Lesson extension in §5.17" |
| 5 | Open Access | Claims MIT licence; repo is Apache-2.0 | Corrected to Apache-2.0 |
| 12 | Licenses | Same MIT / Apache-2.0 mismatch, missing NoRobots licence | Corrected to Apache-2.0; added NoRobots CC-BY-NC-4.0 |
| 13 | Assets | `DATASHEET.md` referenced but does not exist | Removed datasheet claim; model-card + `ARTIFACT.md` only |
| 16 | LLM Usage | "LLMs were not used to draft" contradicts `autoresearch`/`chatgpt_pro_*` artefacts | Softened to an accurate description of review/scaffolding use |

All other items were re-verified against the compiled PDF and stand as
answered. The machine-readable counterpart of this file is
`paper/sections/checklist.tex`, which `\input{}`s cleanly into
`paper/main.tex` in place of the current inline checklist.
