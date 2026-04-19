#!/usr/bin/env python3
from pathlib import Path

submission = Path("reports/final/SUBMISSION_README.md").read_text().lower()
checklist = Path("reports/final/SUBMISSION_CHECKLIST.md").read_text().lower()
issues = []

if "for blind review submissions, submit the anonymized paper source/package" not in submission:
    issues.append("submission_readme_missing_anonymized_package_guidance")
if "exclude the non-anonymous paper source from blind-review bundles" not in submission:
    issues.append("submission_readme_missing_nonanonymous_exclusion_note")
if "anonymous submission" in checklist and "anonymized paper source/package" not in checklist:
    issues.append("checklist_missing_anonymized_package_note")

print(f"METRIC blind_package_issues={len(issues)}")
print("Blind-review package checks passed." if not issues else "\n".join(issues))
