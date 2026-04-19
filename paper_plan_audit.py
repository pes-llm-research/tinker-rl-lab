#!/usr/bin/env python3
"""
Paper Improvement Audit
=======================
Counts unresolved reviewer issues in the GRPO paper.
Each check maps to a specific discovery report concern.
Lower is better (0 = all issues addressed).
"""

import re
import sys


def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def audit_paper() -> dict:
    paper = read_file("reports/final/grpo_agentic_llm_paper.tex")
    appendix = read_file("reports/final/supplementary_appendix.tex")
    bib = read_file("reports/final/references.bib")
    full = paper + "\n" + appendix

    issues = {}

    # === Discovery 1: Evaluation Validity Gaps ===

    # 1. Training-set results not clearly labeled
    # Check if tables/captions contain "training-set" or "training reward" qualifiers
    train_reward_labels = len(re.findall(r"training[- ](?:set|reward|prompt)", full, re.I))
    if train_reward_labels < 4:  # should appear near every results table
        issues["D1_train_label"] = (
            f"Training-set results not consistently labeled (found {train_reward_labels}, need >=4)"
        )

    # 2. Missing comprehensive methods table (compute budgets)
    if not re.search(r"GPU[- ]hours|tokens processed|compute budget|wall[- ]clock", full, re.I):
        issues["D1_compute_budget"] = "No compute budget table (GPU-hours, tokens, wall-clock)"

    # 3. Missing data sizes/splits table
    if not re.search(r"dataset.*size|train.*split.*test|data.*composition", full, re.I):
        issues["D1_data_splits"] = "No explicit dataset sizes/splits documentation"

    # 4. Missing decoding settings documentation
    if not re.search(
        r"(?:decoding|greedy|temperature.*sampling|Decoding.*Protocol|evaluation.*protocol)",
        full,
        re.I,
    ):
        issues["D1_decoding"] = "No decoding settings documented for evaluation"

    # === Discovery 2: Capacity/Exploration/Reward Sparsity ===

    # 5. Group composition analysis missing
    if not re.search(
        r"frac.*(?:all[_-]?bad|all[_-]?good|mixed)|group.*composition|zero[- ](?:reward|advantage).*(?:fraction|rate|percent)",
        full,
        re.I,
    ):
        issues["D2_group_composition"] = "No group-composition analysis (frac_all_bad/good/mixed)"

    # 6. Exploration confounds not discussed quantitatively
    if not re.search(
        r"group.*size.*(?:32|64)|temperature.*(?:sweep|ablat)|curriculum.*(?:learning|strategy)",
        full,
        re.I,
    ):
        issues["D2_exploration_ablations"] = (
            "No discussion of group-size/temperature/curriculum rescue ablations"
        )

    # 7. Missing comparison to RLOO/REINFORCE++
    if "RLOO" not in full and "REINFORCE++" not in full:
        issues["D2_rloo_reinforce"] = "No mention of RLOO/REINFORCE++ baselines"

    # === Discovery 3: Training Stability & MoE Diagnostics ===

    # 8. KL/entropy telemetry not reported for main experiments
    if not re.search(
        r"entropy.*(?:collapse|trajectory|decay)|KL.*(?:trajectory|divergence.*SFT|anchor)",
        full,
        re.I,
    ):
        issues["D3_kl_entropy"] = "No KL/entropy telemetry reported for main experiments"

    # 9. MoE routing diagnostics missing
    if not re.search(
        r"router.*(?:entropy|shift|metric)|expert.*(?:load|balance|utiliz)", full, re.I
    ):
        issues["D3_moe_routing"] = "No MoE routing diagnostics (router entropy, expert load)"

    # 10. Zero-loss step analysis not quantified
    if not re.search(r"zero[- ]loss.*(?:\d+%|fraction|rate)|skipped.*update", full, re.I):
        issues["D3_zero_loss"] = "Zero-loss step frequency not quantified across models"

    # === Discovery 4: PEFT Baseline Positioning ===

    # 11. Missing ToolRM/FC-RewardBench references
    if "ToolRM" not in full and "FC-RewardBench" not in full:
        issues["D4_tool_benchmarks"] = "No reference to ToolRM/FC-RewardBench"

    # 12. Missing S-GRPO/StepGRPO references
    if "S-GRPO" not in full and "StepGRPO" not in full and "step-wise GRPO" not in full.lower():
        issues["D4_step_grpo"] = "No reference to step-wise GRPO variants"

    # 13. Missing DPO/Step-DPO as baselines
    if not re.search(r"DPO.*baseline|compare.*DPO|DPO.*comparison", full, re.I):
        issues["D4_dpo_baseline"] = "DPO not positioned as comparison baseline"

    # 14. QR-Adaptor/LoTA-QAF not discussed
    if "QR-Adaptor" not in full and "LoTA" not in full and "QR-LoRA" not in full:
        issues["D4_peft_context"] = "No discussion of advanced PEFT methods (QR-Adaptor/LoTA-QAF)"

    # 15. Missing future experiments section with standardized eval plan
    if not re.search(
        r"(?:future|planned).*(?:experiment|evaluation|ablation).*(?:plan|roadmap|protocol)",
        full,
        re.I,
    ):
        issues["D_future_plan"] = "No structured future experiments/evaluation plan"

    # === Cross-cutting ===

    # 16. Related work too thin
    related_work_match = re.search(r"\\section\{Related Work\}(.*?)\\section", full, re.S)
    if related_work_match:
        rw_text = related_work_match.group(1)
        rw_cites = len(re.findall(r"\\cite[tp]?\{", rw_text))
        if rw_cites < 8:
            issues["X_related_work_thin"] = f"Related work has only {rw_cites} citations (need >=8)"

    # 17. References too few
    bib_entries = len(re.findall(r"@\w+\{", bib))
    if bib_entries < 15:
        issues["X_bib_entries"] = f"Bibliography has only {bib_entries} entries (need >=15)"

    # === Tier 2: Deeper quality issues from discovery report ===

    # 18. Abstract should explicitly say "training-set reward" not "accuracy"
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", full, re.S)
    if abstract_match:
        abstract_text = abstract_match.group(1)
        if (
            "training-set" not in abstract_text.lower()
            and "training reward" not in abstract_text.lower()
        ):
            issues["T2_abstract_label"] = (
                "Abstract doesn't explicitly label results as training-set metrics"
            )

    # 19. Conclusion should acknowledge scope limitations
    conclusion_match = re.search(
        r"\\section\{Conclusion\}(.*?)(?:\\section|\\end\{document\})", full, re.S
    )
    if conclusion_match:
        conc_text = conclusion_match.group(1)
        if "held-out" not in conc_text.lower() and "test set" not in conc_text.lower():
            issues["T2_conclusion_heldout"] = (
                "Conclusion doesn't mention need for held-out evaluation"
            )

    # 20. Reward function limitations discussed
    if not re.search(
        r"reward.*(?:coarse|hack|limitation)|argument[- ]value.*(?:check|correct)", full, re.I
    ):
        issues["T2_reward_limitations"] = (
            "Reward function limitations (coarseness, hacking risk) not discussed"
        )

    # 21. Safety considerations for tool execution
    if not re.search(r"safety.*(?:tool|execution|loop)|incorrect.*tool.*execution", full, re.I):
        issues["T2_safety"] = "No safety discussion for tool execution risks"

    # 22. Failure taxonomy for tool-calling errors
    if not re.search(
        r"(?:wrong.*function|argument.*mismatch|value.*error).*(?:taxonomy|categor)", full, re.I
    ):
        if not re.search(r"failure.*(?:mode|taxonomy|categor)|error.*(?:type|categor)", full, re.I):
            issues["T2_failure_taxonomy"] = "No failure taxonomy for tool-calling errors"

    # 23. Synthetic-to-real gap quantified with schema characteristics
    if not re.search(
        r"schema.*(?:divers|characteristic|complex)|(?:5|five).*tool.*(?:60|sixty).*(?:k|thousand)",
        full,
        re.I,
    ):
        issues["T2_schema_gap"] = "Synthetic-to-real gap not quantified with schema characteristics"

    # 24. Two-phase learning validated with multiple seeds/tasks
    if not re.search(
        r"two[- ]phase.*(?:seed|task|replic)|phase.*(?:format|reasoning).*(?:seed|replicate)",
        full,
        re.I,
    ):
        issues["T2_two_phase_validation"] = "Two-phase learning not validated across seeds/tasks"

    # 25. Appendix diagnostics table present
    if not re.search(r"Entropy.*KL.*(?:table|diagnostic)|diagnostic.*telemetry", appendix, re.I):
        issues["T2_diagnostics_table"] = "Appendix missing diagnostics table with entropy/KL data"

    # === Tier 3: Questions for Authors / deeper rigor ===

    # 26. Tool-calling scoring protocol explained
    if not re.search(
        r"(?:custom|hand[- ]authored).*(?:rubric|scenario)|scoring.*(?:protocol|rubric)", full, re.I
    ):
        issues["T3_scoring_protocol"] = (
            "Tool-calling scoring protocol not explained (rubric, judge type)"
        )

    # 27. SFT baseline 0% JSON validity explained
    if not re.search(
        r"SFT.*(?:plain text|never.*tool|0\\?%.*JSON|lacked.*capacity|defaulted)", full, re.I
    ):
        issues["T3_sft_baseline"] = "SFT 0% JSON validity not explained (format, decoding, parser)"

    # 28. 50-problem subset rationale
    if not re.search(r"50[- ]problem.*(?:subset|reason|chose|limit)", full, re.I):
        issues["T3_humaneval_subset"] = "50-problem HumanEval subset choice not explained"

    # 29. Exact training prompts per domain documented
    if not re.search(r"(?:prompt|template).*(?:format|example)|chat.*template", full, re.I):
        issues["T3_prompt_format"] = "Exact prompt templates not documented per domain"

    # 30. Repeated tool-call penalty mechanism explained
    if not re.search(r"-0\.30.*(?:penalty|repeated)|penalty.*repeated.*tool", full, re.I):
        issues["T3_penalty_mechanism"] = "Repeated tool-call penalty mechanism not explained"

    return issues


def main():
    issues = audit_paper()
    n = len(issues)
    total_checks = 30

    print(f"METRIC reviewer_issues={n}")
    print(f"METRIC total_checks={total_checks}")
    print(f"METRIC resolved={total_checks - n}")

    if issues:
        print(f"\n--- {n} UNRESOLVED ISSUES ---")
        for k, v in sorted(issues.items()):
            print(f"  [{k}] {v}")
    else:
        print(f"\nAll {total_checks} reviewer issues resolved!")

    return n


if __name__ == "__main__":
    sys.exit(main())
