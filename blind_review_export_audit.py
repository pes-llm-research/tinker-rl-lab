#!/usr/bin/env python3
from pathlib import Path

script = Path("reports/final/prepare_blind_review_package.py")
submission = Path("reports/final/SUBMISSION_README.md").read_text().lower()
checklist = Path("reports/final/SUBMISSION_CHECKLIST.md").read_text().lower()
issues = []

if not script.exists():
    issues.append("missing_blind_review_export_script")
else:
    text = script.read_text().lower()
    if "grpo_agentic_llm_paper_anonymous.tex" not in text:
        issues.append("export_script_missing_anonymized_tex")
    if "grpo_agentic_llm_paper.tex" in text:
        issues.append("export_script_mentions_nonanonymous_tex")
    if "compiled pdfs" not in text and ".pdf" not in text:
        issues.append("export_script_missing_build_artifact_exclusion_note")

if "prepare_blind_review_package.py" not in submission:
    issues.append("submission_readme_missing_export_script_reference")
if "prepare_blind_review_package.py" not in checklist:
    issues.append("submission_checklist_missing_export_script_reference")

print(f"METRIC export_issues={len(issues)}")
print("Blind-review export checks passed." if not issues else "\n".join(issues))
