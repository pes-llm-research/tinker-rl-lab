#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = (ROOT / "reports/final/grpo_agentic_llm_paper.tex").read_text(encoding="utf-8").lower()
REPORT = (ROOT / "reports/final/capstone_final_report.md").read_text(encoding="utf-8").lower()
SUPP = (ROOT / "reports/final/supplementary_appendix.tex").read_text(encoding="utf-8").lower()

issues = []


def need(cond, code, msg):
    if not cond:
        issues.append((code, msg))


need(
    "held-out gsm8k evaluation is still pending" in PAPER or "held-out test evaluation" in PAPER,
    "heldout_scope",
    "Main paper should explicitly state that held-out GSM8K evaluation is still pending and math claims are not generalization claims.",
)
need(
    ("kl" in PAPER and "entropy" in PAPER and "limitation" in PAPER)
    or ("kl regularization" in SUPP and "entropy" in SUPP),
    "kl_entropy_limit",
    "Paper/supplement should discuss lack of KL anchoring or entropy diagnostics as a limitation and future mitigation.",
)
need(
    ("custom reward-derived scenario scores" in PAPER)
    or ("inter-rater reliability was not measured" in PAPER)
    or ("custom metrics" in SUPP and "inter-rater reliability" in SUPP),
    "tool_eval_protocol",
    "Tool-calling section should clarify that the multi-turn scores are custom scenario/reward scores and that inter-rater reliability was not measured.",
)
need(
    ("50-problem subset" in PAPER or "50-item subset" in PAPER)
    and (
        "full humaneval" in PAPER or "standard harness" in PAPER or "non-standard subset" in PAPER
    ),
    "codegen_subset",
    "Paper should clearly disclose that code generation used a 50-problem subset rather than the full standard HumanEval harness and avoid significance claims.",
)
need(
    ("exploration" in PAPER and "group size" in PAPER and "temperature" in PAPER)
    or ("capacity threshold" in SUPP and "exploration" in SUPP),
    "capacity_confound",
    "Capacity-threshold discussion should explicitly acknowledge exploration/reward-sparsity confounds and name the missing ablations (group size, temperature, curriculum).",
)
need(
    ("routing entropy" in PAPER)
    or ("expert load imbalance" in SUPP)
    or ("routing diagnostics" in PAPER)
    or ("routing entropy" in SUPP),
    "moe_diagnostics",
    "MoE discussion should explicitly say that routing entropy / expert-load diagnostics are missing and are future work.",
)
need(
    ("compute budget" in PAPER or "gpu-hours" in PAPER or "tokens processed" in PAPER)
    and ("data split" in PAPER or "splits" in SUPP),
    "budget_and_splits",
    "Paper/supplement should explicitly summarize compute budgets and data split limitations.",
)
need(
    (
        "toolrm" in PAPER
        or "fc-rewardbench" in PAPER
        or "rloo" in PAPER
        or "reinforce++" in PAPER
        or "s-grpo" in PAPER
        or "proxy state" in PAPER
        or "qlora" in PAPER
    ),
    "related_work_positioning",
    "Paper should explicitly position itself against missing evaluation/baseline families (e.g. ToolRM, FC-RewardBench, proxy-state evaluation, RLOO/REINFORCE++, S-GRPO, QLoRA context).",
)
need(
    "release code" in PAPER or "release code" in REPORT or "evaluation scripts" in SUPP,
    "replication_release",
    "Report should explicitly state what code/prompts/evaluation assets are or are not released for replication.",
)
need(
    "near-perfect accuracy" not in PAPER,
    "paper.near_perfect_overclaim",
    "Main paper should not describe GSM8K training-set reward as near-perfect accuracy.",
)
need(
    "confirming the threshold" not in REPORT and "confirms the threshold" not in REPORT,
    "report.threshold_overclaim",
    "Capstone report should avoid saying a single-seed 4B result confirms the capacity threshold.",
)
need(
    "8b: 100% peak on gsm8k" not in REPORT,
    "report.peak_table_overclaim",
    "Summary tables should not present peak GSM8K training-step numbers in a way that reads like benchmark performance.",
)

print(f"METRIC caveat_issues={len(issues)}")
for code, msg in issues:
    print(f"ISSUE {code}: {msg}")
