#!/usr/bin/env python3
from pathlib import Path

root = Path("reports/final")
issues = []

# Build artifacts should not sit in the review package directory.
for name in [
    "grpo_agentic_llm_paper_anonymous.aux",
    "grpo_agentic_llm_paper_anonymous.log",
    "grpo_agentic_llm_paper_anonymous.out",
    "grpo_agentic_llm_paper_anonymous.pdf",
]:
    if (root / name).exists():
        issues.append(f"build_artifact_present:{name}")

submission = (root / "SUBMISSION_README.md").read_text().lower()
if "fresh clone" not in submission:
    issues.append("submission_readme_missing_clean_export_guidance")
if "do not include generated build artifacts" not in submission:
    issues.append("submission_readme_missing_build_artifact_exclusion_note")

print(f"METRIC package_issues={len(issues)}")
print("Submission packaging checks passed." if not issues else "\n".join(issues))
