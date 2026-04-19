#!/usr/bin/env python3
import re
from pathlib import Path

readme = Path("reports/final/README.md").read_text()
checklist = Path("reports/final/SUBMISSION_CHECKLIST.md").read_text()
submission = Path("reports/final/SUBMISSION_README.md").read_text()
ideas = Path("autoresearch.ideas.md").read_text() if Path("autoresearch.ideas.md").exists() else ""

issues = []

# README should clearly label non-standardized headline metrics.
if "## Key Results (Training-Set)" not in readme:
    issues.append("readme_missing_key_results_scope_header")
if "tool results are internal/custom" not in readme.lower():
    issues.append("readme_missing_tool_custom_caveat")
if "50-problem" not in readme.lower() or "subset" not in readme.lower():
    issues.append("readme_missing_humaneval_subset_caveat")
if "held-out" not in readme.lower():
    issues.append("readme_missing_heldout_language")

# Checklist should not present headline results without caveats.
if "Key Results to Highlight" in checklist and "preliminary / custom" not in checklist.lower():
    issues.append("checklist_missing_preliminary_key_results_label")
if "50-problem subset" not in checklist.lower():
    issues.append("checklist_missing_humaneval_subset_note")
if "custom internal" not in checklist.lower() and "custom judge-derived" not in checklist.lower():
    issues.append("checklist_missing_tool_custom_note")
if "training-set reward" not in checklist.lower():
    issues.append("checklist_missing_training_set_math_note")
if "checkpoints available" in checklist.lower() or "checkpoints available" in readme.lower():
    issues.append("misleading_checkpoint_availability_claim")

# Submission README should also preserve caveats and avoid completion claims.
if "Key Results Summary" in submission and "preliminary / custom" not in submission.lower():
    issues.append("submission_missing_preliminary_label")
if "50-problem subset" not in submission.lower():
    issues.append("submission_missing_humaneval_subset_note")
if "custom internal" not in submission.lower() and "custom judge-derived" not in submission.lower():
    issues.append("submission_missing_tool_custom_note")
if "training-set reward" not in submission.lower():
    issues.append("submission_missing_training_set_math_note")
if "model checkpoint urls" in submission.lower():
    issues.append("submission_overstates_checkpoint_release")
if re.search(r"\[x\].*9-page limit satisfied", submission):
    issues.append("submission_has_unverified_page_count_checkbox")

# Ideas backlog should stay focused on unrun, concrete experiment paths.
if "paper-improvement roadmap" in ideas.lower() or "plan audit" in ideas.lower():
    issues.append("ideas_contains_already_done_meta_work")

print(f"METRIC claim_issues={len(issues)}")
print("All submission claim checks passed." if not issues else "\n".join(issues))
