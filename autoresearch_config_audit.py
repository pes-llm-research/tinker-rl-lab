#!/usr/bin/env python3
from pathlib import Path

md = Path("autoresearch.md").read_text().lower()
issues = []

if "suite_issues" not in md:
    issues.append("autoresearch_md_missing_suite_metric")
if "run_all_audits.py" not in md:
    issues.append("autoresearch_md_missing_unified_suite_reference")
if "reviewer_issues" not in md:
    issues.append("autoresearch_md_missing_primary_reviewer_metric_context")

print(f"METRIC config_issues={len(issues)}")
print("Autoresearch config checks passed." if not issues else "\n".join(issues))
