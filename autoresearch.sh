#!/bin/bash
set -euo pipefail

# Paper Acceptance Benchmark
# Measures how well the repo addresses NeurIPS reviewer concerns.

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

echo "METRIC acceptance_score=$SCORE"
echo "METRIC max_acceptance_score=$MAX_SCORE"
echo "METRIC weaknesses_addressed=$ADDRESSED"
echo "METRIC weakness_total=$WEAKNESS_COUNT"
echo "METRIC script_bonus=$BONUS"
echo "METRIC ai_scientist_runs=$AI_SCIENTIST_RUNS"
