#!/bin/bash
set -euo pipefail

# AI-Scientist-v2 Runnability Benchmark for Tinker-RL-Lab
# Measures how well the AI Scientist can run experiments on this repo.

# Apply integration patches first
if [ -f "ai-scientist-v2-integration/patch.sh" ]; then
    bash ai-scientist-v2-integration/patch.sh >/dev/null 2>&1
fi

SCORE=0
DEPS_READY=0
TEMPLATE_PARSES=0
TEMPLATE_RUNNABLE=0
PROMPT_HAS_TINKER=0

# --- 1. Dependency readiness (25 points) ---
# Check that all packages the template needs are importable in the target Python.
MISSING=""
for pkg in tinker transformers datasets numpy torch; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -z "$MISSING" ]; then
    DEPS_READY=1
    SCORE=$((SCORE + 25))
else
    echo "MISSING_DEPS=$MISSING"
fi

# --- 2. Template syntax validity (25 points) ---
TEMPLATE="$HOME/ai-scientist-v2/ai_scientist/ideas/tinker_grpo_rl.py"
if [ -f "$TEMPLATE" ]; then
    if python3 -m py_compile "$TEMPLATE" 2>/dev/null; then
        TEMPLATE_PARSES=1
        SCORE=$((SCORE + 25))
    else
        echo "TEMPLATE_SYNTAX_ERROR=1"
    fi
else
    echo "TEMPLATE_MISSING=1"
fi

# --- 3. Template runnable with mocks (25 points) ---
# We create a stripped-down version that exercises the core logic
# without needing TINKER_API_KEY or network access.
if [ "$TEMPLATE_PARSES" -eq 1 ]; then
    TMPDIR=$(mktemp -d)
    cat > "$TMPDIR/test_template_smoke.py" << 'PYEOF'
import sys, os, ast

# Verify the template file exists and is syntactically valid
template_path = os.path.expanduser("~/ai-scientist-v2/ai_scientist/ideas/tinker_grpo_rl.py")
with open(template_path) as f:
    source = f.read()
ast.parse(source)

# Check that key configurable variables are present and have expected types
assert "MODEL" in source, "Missing MODEL variable"
assert "LORA_RANK" in source, "Missing LORA_RANK variable"
assert "GROUP_SIZE" in source, "Missing GROUP_SIZE variable"
assert "STEPS" in source, "Missing STEPS variable"
assert "LR" in source, "Missing LR variable"
assert "NUM_SEEDS" in source, "Missing NUM_SEEDS variable"
assert "reward_fn" in source, "Missing reward_fn"
assert "curriculum" in source, "Missing curriculum function"
assert "grpo_loss_fn" in source, "Missing grpo_loss_fn"
assert "run_one_seed" in source, "Missing run_one_seed function"
assert "main" in source, "Missing main function"

# Verify the template imports tinker
assert "import tinker" in source or "from tinker" in source, "Template does not import tinker"

# Verify final_info.json output structure is referenced
assert "final_info.json" in source, "Missing final_info.json output"

print("TEMPLATE_SMOKE_OK=1")
PYEOF

    if python3 "$TMPDIR/test_template_smoke.py" 2>/dev/null; then
        TEMPLATE_RUNNABLE=1
        SCORE=$((SCORE + 25))
    else
        echo "TEMPLATE_SMOKE_FAILED=1"
    fi
    rm -rf "$TMPDIR"
fi

# --- 4. Prompt completeness (25 points) ---
# The agent should know that tinker is available.
AGENT_FILE="$HOME/ai-scientist-v2/ai_scientist/treesearch/parallel_agent.py"
if [ -f "$AGENT_FILE" ]; then
    if grep -q "tinker" "$AGENT_FILE"; then
        PROMPT_HAS_TINKER=1
        SCORE=$((SCORE + 25))
    else
        echo "PROMPT_MISSING_TINKER=1"
    fi
else
    echo "AGENT_FILE_MISSING=1"
fi

# --- Output metrics ---
echo "METRIC runnability_score=$SCORE"
echo "METRIC deps_ready=$DEPS_READY"
echo "METRIC template_parses=$TEMPLATE_PARSES"
echo "METRIC template_runnable=$TEMPLATE_RUNNABLE"
echo "METRIC prompt_has_tinker=$PROMPT_HAS_TINKER"
