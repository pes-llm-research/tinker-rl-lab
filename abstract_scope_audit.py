#!/usr/bin/env python3
import re
from pathlib import Path

files = {
    "main_tex": Path("reports/final/grpo_agentic_llm_paper.tex").read_text().lower(),
    "anon_tex": Path("reports/final/grpo_agentic_llm_paper_anonymous.tex").read_text().lower(),
    "markdown": Path("reports/final/grpo_agentic_llm_paper.md").read_text().lower(),
}
issues = []

for name, text in files.items():
    if name == "markdown":
        abstract = text.split("## 1. introduction")[0]
    else:
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
        abstract = m.group(1) if m else text[:2000]

    if "custom" not in abstract:
        issues.append(f"{name}_abstract_missing_custom_eval_caveat")
    if "50-problem subset" not in abstract:
        issues.append(f"{name}_abstract_missing_humaneval_subset_caveat")
    if "training-set reward" not in abstract:
        issues.append(f"{name}_abstract_missing_training_reward_caveat")
    if "held-out" not in abstract:
        issues.append(f"{name}_abstract_missing_heldout_caveat")

print(f"METRIC abstract_issues={len(issues)}")
print("All abstract scope checks passed." if not issues else "\n".join(issues))
