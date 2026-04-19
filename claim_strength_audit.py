#!/usr/bin/env python3
from pathlib import Path

files = {
    "paper_tex": Path("reports/final/grpo_agentic_llm_paper.tex").read_text().lower(),
    "paper_md": Path("reports/final/grpo_agentic_llm_paper.md").read_text().lower(),
    "paper_anon": Path("reports/final/grpo_agentic_llm_paper_anonymous.tex").read_text().lower(),
    "capstone": Path("reports/final/capstone_final_report.md").read_text().lower(),
}
issues = []

for name, text in files.items():
    if "can reliably improve small language models" in text:
        issues.append(f"{name}_uses_reliably_improve_opening_claim")
    if "can reliably optimize reward on verifiable tasks" in text:
        issues.append(f"{name}_uses_reliably_optimize_opening_claim")
    if "practical, compute-efficient method" in text:
        issues.append(f"{name}_uses_broad_compute_efficient_claim")
    if (
        "strong gains on task-specific metrics" in text
        and "custom internal evaluation protocol" in text
    ):
        # okay; keep pass
        pass
    elif "strong gains on task-specific metrics" in text:
        issues.append(f"{name}_strong_gains_without_nearby_scope_qualifier")

print(f"METRIC strength_issues={len(issues)}")
print("Claim-strength checks passed." if not issues else "\n".join(issues))
