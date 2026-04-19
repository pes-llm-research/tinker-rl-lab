#!/usr/bin/env python3
from pathlib import Path

submission = Path("reports/final/SUBMISSION_README.md").read_text().lower()
issues = []

if "python run_all_audits.py" not in submission:
    issues.append("submission_readme_missing_audit_suite_step")
if "do not include generated build artifacts" not in submission:
    issues.append("submission_readme_missing_build_artifact_exclusion_note")
if (
    "remove this section or replace it with the venue's anonymized contact mechanism"
    not in submission
):
    issues.append("submission_readme_missing_blind_review_contact_note")

print(f"METRIC workflow_issues={len(issues)}")
print("Submission workflow checks passed." if not issues else "\n".join(issues))
