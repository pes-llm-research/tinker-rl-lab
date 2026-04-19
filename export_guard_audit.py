#!/usr/bin/env python3
from pathlib import Path

script = Path("reports/final/prepare_blind_review_package.py").read_text().lower()
issues = []

if "run_all_audits.py" not in script:
    issues.append("export_script_missing_audit_guard")
if "--skip-audits" not in script:
    issues.append("export_script_missing_skip_audits_override")
if "subprocess.run" not in script:
    issues.append("export_script_not_invoking_audit_process")

print(f"METRIC export_guard_issues={len(issues)}")
print("Export guard checks passed." if not issues else "\n".join(issues))
