#!/usr/bin/env python3
import re
import subprocess
import sys

AUDITS = [
    "paper_improvement_audit.py",
    "submission_claim_audit.py",
    "paper_sync_audit.py",
    "capstone_claim_audit.py",
    "abstract_scope_audit.py",
    "heldout_readiness_audit.py",
    "anonymization_repro_audit.py",
    "claim_strength_audit.py",
    "submission_package_audit.py",
    "submission_workflow_audit.py",
    "blind_review_package_audit.py",
    "blind_review_export_audit.py",
    "export_guard_audit.py",
]

failures = []
for audit in AUDITS:
    proc = subprocess.run(["python", audit], capture_output=True, text=True)
    out = proc.stdout.strip()
    match = re.search(r"METRIC\s+\w+=(\d+)", out)
    metric = int(match.group(1)) if match else None
    print(f"=== {audit} ===")
    print(out)
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if proc.returncode != 0 or (metric is not None and metric != 0):
        failures.append((audit, proc.returncode, metric))
    print()

print(f"METRIC suite_issues={len(failures)}")
print(f"METRIC audits_total={len(AUDITS)}")
print(f"METRIC audits_passing={len(AUDITS) - len(failures)}")

if failures:
    print("Failing audits:")
    for audit, rc, metric in failures:
        print(f"  - {audit}: rc={rc}, metric={metric}")
    sys.exit(1)

print("All audits passing.")
