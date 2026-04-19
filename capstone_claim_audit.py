#!/usr/bin/env python3
from pathlib import Path

text = Path("reports/final/capstone_final_report.md").read_text().lower()
issues = []

if "50-problem subset" not in text:
    issues.append("missing_humaneval_subset_caveat")
if "custom reward-derived scenario scores" not in text and "custom internal evaluation" not in text:
    issues.append("missing_tool_custom_caveat")
if "training-set reward" not in text:
    issues.append("missing_training_set_math_caveat")
if "held-out" not in text:
    issues.append("missing_heldout_language")
if "rloo / reinforce++ / s-grpo comparison" not in text and "rloo" not in text:
    issues.append("missing_baseline_positioning")
if "reliable tool caller" in text or "reliable tool callers" in text:
    issues.append("has_reliable_tool_caller_overclaim")
if "grpo enables significant capability gains" in text and "custom" not in text[:4000]:
    issues.append("abstract_missing_custom_eval_context")

print(f"METRIC capstone_issues={len(issues)}")
print("All capstone claim checks passed." if not issues else "\n".join(issues))
