#!/usr/bin/env python3
from pathlib import Path

anon = Path("reports/final/grpo_agentic_llm_paper_anonymous.tex").read_text().lower()
supp = Path("reports/final/supplementary_appendix.tex").read_text().lower()
submission = Path("reports/final/SUBMISSION_README.md").read_text().lower()

issues = []

# Blind-review hygiene.
if "pes university" in submission or "mtech dsai" in submission:
    issues.append("submission_readme_contains_institution_contact")
if "huggingface.co/madhu2133" in supp:
    issues.append("supplementary_contains_identifying_hf_username")

# Reproducibility language should avoid brittle/local-only paths.
if "/tmp/gsm8k_" in supp or "/tmp/grpo_" in supp:
    issues.append("supplementary_contains_local_tmp_log_paths")
if "model checkpoints are hosted on tinker and huggingface hub" in supp:
    issues.append("supplementary_overstates_checkpoint_hosting_availability")

# Anonymous paper should stay anonymous.
if "anonymous institution" not in anon:
    issues.append("anonymous_paper_missing_anonymous_institution_marker")
if "acknowledgments" in anon and "pes university" in anon:
    issues.append("anonymous_paper_contains_institution_in_acknowledgments")

print(f"METRIC anon_issues={len(issues)}")
print("Anonymization/reproducibility checks passed." if not issues else "\n".join(issues))
