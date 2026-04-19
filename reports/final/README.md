# Reports - Final Submission

This directory contains the final capstone report and conference paper for the GRPO Agentic LLM Fine-Tuning project.

## ⚠️ CRITICAL: Standardized Evaluation Still Required

**The paper now acknowledges its evaluation-scope limitations, but key reviewer-facing gaps still need to be closed.** The strongest current evidence is in training-set reward optimization and training dynamics, not yet in fully standardized held-out generalization across math, tool calling, and code generation.

See `PAPER_IMPROVEMENT_PLAN.md` for the concrete remediation roadmap.

### To Complete the Paper (A-grade path):

Run the held-out GSM8K test evaluation:

```bash
# With Tinker (if checkpoint still available)
TINKER_API_KEY=your_key python evaluate_gsm8k_test.py \
    --use_tinker \
    --run_id 5db4e965 \
    --output gsm8k_test_results.json

# With local model (requires GPU)
python evaluate_gsm8k_test.py \
    --model_name Qwen/Qwen3-8B \
    --output gsm8k_test_results.json
```

If results show >40% accuracy on held-out test, update Section 4.3.3 with actual numbers. This will transform the paper from a training dynamics study to a true generalization claim.

## Files

### Improvement Plan
- `PAPER_IMPROVEMENT_PLAN.md` - prioritized plan to address reviewer concerns, standardize evaluation, and strengthen the paper

### Capstone Report
- `capstone_final_report.md` - Full capstone report (honest about limitations)
- `capstone_final_report.docx` - Word version
- `group6_final_report.pdf` - Canonical integrated final report with paper findings distributed across chapters
- `group6_final_report.tex` - LaTeX export of the canonical integrated final report
- `group6_final_report.docx` - Word export of the canonical integrated final report
- `build_group6_final_report.sh` - Rebuilds the integrated PDF, TeX, and DOCX from `capstone_final_report.md`

### Legacy Wrapper
- `group6_final_report_with_appended_paper.tex` - Legacy wrapper that embeds `group6.pdf` and `paper/main.pdf`; kept for traceability, not the recommended final submission artifact

### Conference Paper (NeurIPS/ICML Format)
- `grpo_agentic_llm_paper.tex` - LaTeX source
- `grpo_agentic_llm_paper.md` - Markdown version
- `grpo_agentic_llm_paper_anonymous.tex` - Anonymized for blind review
- `references.bib` - Bibliography
- `nips_style.sty` - NeurIPS/ICML style

### Evaluation
- `evaluate_gsm8k_test.py` - Held-out GSM8K evaluation script (Tinker API or local HF)
- `run_heldout_parallel.sh` - Run 5-seed parallel evaluation
- `supplementary_appendix.tex` - Additional experimental details

### Reproducibility Notebooks
- `../../submission_colab.ipynb` - Standard GRPO training + evaluation Colab
- `../../advanced_rl_colab.ipynb` - Dr. GRPO, DAPO, DPO Colab (advanced algorithms)

## Key Results (Training-Set)

These numbers should not all be read as standardized benchmark claims:
- **Tool results are internal/custom** and still need standardized evaluator disclosure or replacement.
- **HumanEval is currently a 50-problem subset result**, not yet the canonical full-harness benchmark.
- **Math is the strongest current evidence**, but the main reported GRPO math numbers are still training-set reward metrics until full held-out evaluation is completed.

| Task | Before | After | Scope note |
|------|--------|-------|------------|
| JSON Tool Calls | 0% | 92% | custom internal tool-calling setup |
| Multi-turn Quality | 0.72 | 0.91 | custom judge-derived internal scenario score |
| HumanEval Pass@1 | 32% | 40% | preliminary 50-problem subset |
| GSM8K Train Reward | - | 30.0% ± 2.5% | training-set reward, not held-out test accuracy |

## Paper Status

✅ **Completed**: Honest limitation disclosure  
✅ **Completed**: W&B logging (17 Tinker runs uploaded to tinker-rl-scaling project)  
✅ **Completed**: Advanced RL notebook (Dr. GRPO, DAPO, DPO) -- `advanced_rl_colab.ipynb`  
✅ **Completed**: Submission Colab -- `submission_colab.ipynb`  
✅ **Completed**: All 13 audits passing (0 issues)  
🔄 **In Progress**: Held-out GSM8K evaluation (5 seeds x 200 examples via Tinker API)  
⚠️ **Pending**: Standardized tool-calling evaluation / judge protocol disclosure  
⚠️ **Pending**: Canonical full HumanEval/MBPP evaluation  
⚠️ **Pending**: Reproducibility packaging for prompts, schemas, and checkpoints

## Highest-Leverage Next Step

Run `evaluate_gsm8k_test.py` on the trained checkpoint and update Section 4.3.3 with full held-out results. That is the single highest-leverage improvement, but not the only one: the paper also needs standardized tool/code evaluation or narrower claim boundaries. See `PAPER_IMPROVEMENT_PLAN.md` for the full sequencing.

## Audit Suite

Before submission or export, run:

```bash
python run_all_audits.py
```

This verifies the paper, capstone, submission docs, anonymization hygiene, held-out-evaluation readiness, claim strength, packaging checks, and submission-workflow checks in one pass.

For blind-review export, you can generate a clean bundle with:

```bash
python reports/final/prepare_blind_review_package.py --force
```

By default, the export script runs `python run_all_audits.py` first and refuses to package files if the audit suite is failing.

## Authors

Arvind C R, Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L, Dhruva N Murthy, Mohammad Rafi  
Group 6, MTech DSAI, PES University
