#!/usr/bin/env python3
from pathlib import Path

script = Path("reports/final/evaluate_gsm8k_test.py").read_text().lower()
readme = Path("reports/final/README.md").read_text().lower()
capstone = Path("reports/final/capstone_final_report.md").read_text().lower()
paper = Path("reports/final/grpo_agentic_llm_paper.tex").read_text().lower()

issues = []

# Evaluation script should compute CI / preserve deterministic protocol metadata.
if "bootstrap" not in script and "confidence_interval" not in script and "ci_" not in script:
    issues.append("eval_script_missing_confidence_interval")
if "dataset_split" not in script or "temperature" not in script or "seed" not in script:
    issues.append("eval_script_missing_protocol_metadata")
if 'choices=["test"]' not in script and "choices=['test']" not in script:
    issues.append("eval_script_not_locked_to_test_split")

# Docs should not imply universal checkpoint availability.
if "all training runs produce tinker-hosted model checkpoints" in capstone:
    issues.append("capstone_overstates_checkpoint_availability")
if "all runs produce tinker-hosted checkpoints" in paper:
    issues.append("paper_overstates_checkpoint_availability")
if "all tinker training runs, logs, and model checkpoints are available" in capstone:
    issues.append("capstone_claims_all_checkpoints_available")

# README should frame checkpoint access conditionally.
if "if checkpoint still available" not in readme:
    issues.append("readme_missing_conditional_checkpoint_access_language")

print(f"METRIC readiness_issues={len(issues)}")
print("Held-out evaluation readiness checks passed." if not issues else "\n".join(issues))
