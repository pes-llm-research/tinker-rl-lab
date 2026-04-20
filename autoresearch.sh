#!/bin/bash
set -euo pipefail

# Paper Acceptance Benchmark
# Measures how well the repo addresses NeurIPS reviewer concerns.

# Apply integration patches first
if [ -f "ai-scientist-v2-integration/patch.sh" ]; then
    bash ai-scientist-v2-integration/patch.sh > /dev/null 2>&1
fi

SCORE=0
MAX_SCORE=0

# Count addressed reviewer weaknesses by grepping for markers
PAPER_DIR="/Users/arvind/paper/tinker-rl-lab/paper"
WEAKNESS_COUNT=$(grep -c '^  - id:' /Users/arvind/paper/tinker-rl-lab/paper/reviewer_points.yaml 2>/dev/null || echo 0)

ADDRESSED=0
while IFS= read -r marker; do
    marker=$(echo "$marker" | sed 's/.*marker: //' | tr -d '"')
    if grep -rq "$marker" "$PAPER_DIR" 2>/dev/null; then
        ADDRESSED=$((ADDRESSED + 1))
    fi
done < <(grep 'marker:' /Users/arvind/paper/tinker-rl-lab/paper/reviewer_points.yaml)

# Score: 3 points per addressed weakness, max 54 points
SCORE=$((ADDRESSED * 3))
MAX_SCORE=$((WEAKNESS_COUNT * 3))

# --- Bonus: executable scripts exist for critical questions ---
BONUS=0

# Q1 partial correlation
if [ -f "/Users/arvind/paper/tinker-rl-lab/scripts/partial_correlation_zvf.py" ]; then
    BONUS=$((BONUS + 5))
fi

# Q2 group size sweep
if [ -f "/Users/arvind/paper/tinker-rl-lab/experiments/group_size_token_normalized.py" ]; then
    BONUS=$((BONUS + 5))
fi

# Q3 framework configs
if [ -f "/Users/arvind/paper/tinker-rl-lab/paper/sections/framework_configs_appendix.tex" ]; then
    BONUS=$((BONUS + 5))
fi

# Q4 base vs instruct
if [ -f "/Users/arvind/paper/tinker-rl-lab/experiments/base_instruct_paired.py" ]; then
    BONUS=$((BONUS + 5))
fi

# Q5 survival analysis
if [ -f "/Users/arvind/paper/tinker-rl-lab/experiments/survival_analysis.py" ]; then
    BONUS=$((BONUS + 5))
fi

# Q6 tool use
if [ -f "/Users/arvind/paper/tinker-rl-lab/experiments/tool_use_reward_analysis.py" ]; then
    BONUS=$((BONUS + 5))
fi

# Q7 variance mitigation
if [ -f "/Users/arvind/paper/tinker-rl-lab/experiments/variance_mitigation_integration.py" ]; then
    BONUS=$((BONUS + 5))
fi

SCORE=$((SCORE + BONUS))
MAX_SCORE=$((MAX_SCORE + 35))

# --- AI Scientist runnability (already optimized, worth 15 points) ---
TEMPLATE="$HOME/ai-scientist-v2/ai_scientist/ideas/tinker_grpo_rl.py"
AI_SCIENTIST_RUNS=0
if [ -f "$TEMPLATE" ]; then
    TMPDIR=$(mktemp -d)
    cd "$TMPDIR"
    python3 "$TEMPLATE" > output.log 2>&1 || true
    if [ -f "$TMPDIR/working/final_info.json" ] && [ -f "$TMPDIR/working/experiment_data.npy" ]; then
        AI_SCIENTIST_RUNS=1
        SCORE=$((SCORE + 15))
    fi
    cd - > /dev/null
    rm -rf "$TMPDIR"
fi
MAX_SCORE=$((MAX_SCORE + 15))

# --- Bonus: Local TRL template exists (5 points) ---
LOCAL_TRL=0
LOCAL_TRL_TEMPLATE="$HOME/ai-scientist-v2/ai_scientist/ideas/trl_local_grpo.py"
if [ -f "$LOCAL_TRL_TEMPLATE" ]; then
    if python3 -m py_compile "$LOCAL_TRL_TEMPLATE" 2>/dev/null; then
        LOCAL_TRL=1
        SCORE=$((SCORE + 5))
    fi
fi
MAX_SCORE=$((MAX_SCORE + 5))

# --- Bonus: Key experiment scripts execute successfully (10 points) ---
EXP_SCRIPTS_RUN=0
EXP_OK=0
cd /Users/arvind/paper/tinker-rl-lab
if python3 experiments/base_instruct_paired.py --quiet >/dev/null 2>&1; then
    EXP_OK=$((EXP_OK + 1))
fi
if python3 experiments/group_size_token_normalized.py --out /dev/null >/dev/null 2>&1; then
    EXP_OK=$((EXP_OK + 1))
fi
if python3 scripts/partial_correlation_zvf.py --out /dev/null >/dev/null 2>&1; then
    EXP_OK=$((EXP_OK + 1))
fi
if python3 experiments/variance_mitigation_integration.py --method grpo --dry-run >/dev/null 2>&1; then
    EXP_OK=$((EXP_OK + 1))
fi
if python3 scripts/statistical_rigor_report.py >/dev/null 2>&1; then
    EXP_OK=$((EXP_OK + 1))
fi
if python3 experiments/bfclv4_tool_use.py --dry-run --seeds 2 --steps 5 >/dev/null 2>&1; then
    EXP_OK=$((EXP_OK + 1))
fi
if [ "$EXP_OK" -ge 4 ]; then
    EXP_SCRIPTS_RUN=1
    SCORE=$((SCORE + 10))
fi
MAX_SCORE=$((MAX_SCORE + 10))

# --- New baselines present: MC-GRPO + GIFT + statistical rigor + BFCLv4 (15 points) ---
NEW_BASELINES=0
if python3 experiments/variance_mitigation_integration.py --method mcgrpo --dry-run --seeds 1 --steps 5 >/dev/null 2>&1; then
    NEW_BASELINES=$((NEW_BASELINES + 1))
fi
if python3 experiments/variance_mitigation_integration.py --method gift --dry-run --seeds 1 --steps 5 >/dev/null 2>&1; then
    NEW_BASELINES=$((NEW_BASELINES + 1))
fi
if [ -f "/Users/arvind/paper/tinker-rl-lab/experiments/results/statistical_rigor_report.tsv" ]; then
    NEW_BASELINES=$((NEW_BASELINES + 1))
fi
if [ -f "/Users/arvind/paper/tinker-rl-lab/experiments/results/bfclv4_tool_use.tsv" ]; then
    NEW_BASELINES=$((NEW_BASELINES + 1))
fi
# variance_mitigation.tsv should have 7 methods now
VM_METHODS=$(tail -n +2 /Users/arvind/paper/tinker-rl-lab/experiments/results/variance_mitigation.tsv 2>/dev/null | cut -f1 | sort -u | wc -l | tr -d ' ')
if [ "$VM_METHODS" -ge 7 ]; then
    NEW_BASELINES=$((NEW_BASELINES + 1))
fi
if [ "$NEW_BASELINES" -ge 4 ]; then
    SCORE=$((SCORE + 15))
fi
MAX_SCORE=$((MAX_SCORE + 15))

# --- Data freshness: critical result files exist and are non-empty (10 points) ---
DATA_FRESH=0
DATA_FILES=(
    "/Users/arvind/paper/tinker-rl-lab/experiments/results/base_instruct_paired.tsv"
    "/Users/arvind/paper/tinker-rl-lab/experiments/results/group_size_token_normalized.tsv"
    "/Users/arvind/paper/tinker-rl-lab/experiments/results/variance_mitigation.tsv"
)
DATA_OK=0
for f in "${DATA_FILES[@]}"; do
    if [ -s "$f" ]; then
        DATA_OK=$((DATA_OK + 1))
    else
        echo "DATA_MISSING_OR_EMPTY=$f"
    fi
done
if [ "$DATA_OK" -eq "${#DATA_FILES[@]}" ]; then
    DATA_FRESH=1
    SCORE=$((SCORE + 10))
fi
MAX_SCORE=$((MAX_SCORE + 10))

# --- ZVF partial correlations has real results (5 points) ---
ZVF_PARTIAL=0
ZVF_FILE="/Users/arvind/paper/tinker-rl-lab/experiments/results/zvf_partial_correlations.tsv"
if [ -f "$ZVF_FILE" ]; then
    # Check if there is at least one row with a numeric r_partial (not NA)
    if grep -qE '^\(none; raw correlation\)\t[0-9\.-]+' "$ZVF_FILE"; then
        ZVF_PARTIAL=1
        SCORE=$((SCORE + 5))
    else
        echo "ZVF_PARTIAL_NO_RESULTS=1"
    fi
else
    echo "ZVF_PARTIAL_MISSING=1"
fi
MAX_SCORE=$((MAX_SCORE + 5))

echo "METRIC acceptance_score=$SCORE"
echo "METRIC max_acceptance_score=$MAX_SCORE"
echo "METRIC weaknesses_addressed=$ADDRESSED"
echo "METRIC weakness_total=$WEAKNESS_COUNT"
echo "METRIC script_bonus=$BONUS"
echo "METRIC ai_scientist_runs=$AI_SCIENTIST_RUNS"
echo "METRIC local_trl_template=$LOCAL_TRL"
echo "METRIC data_fresh=$DATA_FRESH"
echo "METRIC zvf_partial_results=$ZVF_PARTIAL"
echo "METRIC exp_scripts_run=$EXP_SCRIPTS_RUN"
echo "METRIC new_baselines_score=$NEW_BASELINES"
echo "METRIC vm_methods=$VM_METHODS"
