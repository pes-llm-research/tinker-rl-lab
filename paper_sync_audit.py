#!/usr/bin/env python3
from pathlib import Path

main_tex = Path("reports/final/grpo_agentic_llm_paper.tex").read_text().lower()
main_md = Path("reports/final/grpo_agentic_llm_paper.md").read_text().lower()
anon_tex = Path("reports/final/grpo_agentic_llm_paper_anonymous.tex").read_text().lower()

issues = []
checks = {
    "50-problem subset": "50-problem subset",
    "custom tool evaluation": "custom",
    "training-set reward": "training-set reward",
    "held-out": "held-out",
    "RLOO": "rloo",
    "REINFORCE++": "reinforce++",
    "Step-DPO": "step-dpo",
}

for label, needle in checks.items():
    if needle in main_tex and needle not in main_md:
        issues.append(f"markdown_missing:{label}")
    if needle in main_tex and needle not in anon_tex:
        issues.append(f"anonymous_missing:{label}")

# Stronger phrase check: markdown/anon should not present reliable-tool-caller language if main tex softened it.
for path_name, text in [("markdown", main_md), ("anonymous", anon_tex)]:
    if "reliable tool callers" in text or "reliable tool caller" in text:
        issues.append(f"{path_name}_has_overclaim_reliable_tool_caller")

print(f"METRIC sync_issues={len(issues)}")
print("All paper mirror sync checks passed." if not issues else "\n".join(issues))
