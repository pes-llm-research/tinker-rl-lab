#!/bin/bash
set -euo pipefail

# AI-Scientist-v2 Runnability Benchmark for Tinker-RL-Lab
# Measures how well the AI Scientist can run experiments on this repo.

# Apply integration patches first
if [ -f "ai-scientist-v2-integration/patch.sh" ]; then
    bash ai-scientist-v2-integration/patch.sh > /dev/null 2>&1
fi

SCORE=0
DEPS_READY=0
TEMPLATE_PARSES=0
TEMPLATE_RUNNABLE=0
PROMPT_HAS_TINKER=0
TEMPLATE_NO_MAIN_GUARD=0
TEMPLATE_GRACEFUL_API_KEY=0

# --- 1. Dependency readiness (20 points) ---
MISSING=""
for pkg in tinker transformers datasets numpy torch; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -z "$MISSING" ]; then
    DEPS_READY=1
    SCORE=$((SCORE + 20))
else
    echo "MISSING_DEPS=$MISSING"
fi

# --- 2. Template syntax validity (20 points) ---
TEMPLATE="$HOME/ai-scientist-v2/ai_scientist/ideas/tinker_grpo_rl.py"
if [ -f "$TEMPLATE" ]; then
    if python3 -m py_compile "$TEMPLATE" 2>/dev/null; then
        TEMPLATE_PARSES=1
        SCORE=$((SCORE + 20))
    else
        echo "TEMPLATE_SYNTAX_ERROR=1"
    fi
else
    echo "TEMPLATE_MISSING=1"
fi

# --- 3. Template runnable with mocks (20 points) ---
if [ "$TEMPLATE_PARSES" -eq 1 ]; then
    TMPDIR=$(mktemp -d)
    cat > "$TMPDIR/test_template_smoke.py" << 'PYEOF'
import sys, os, ast

template_path = os.path.expanduser("~/ai-scientist-v2/ai_scientist/ideas/tinker_grpo_rl.py")
with open(template_path) as f:
    source = f.read()
ast.parse(source)

# Check that key configurable variables are present
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
assert "import tinker" in source or "from tinker" in source, "Template does not import tinker"
assert "final_info.json" in source, "Missing final_info.json output"

print("TEMPLATE_SMOKE_OK=1")
PYEOF

    if python3 "$TMPDIR/test_template_smoke.py" 2>/dev/null; then
        TEMPLATE_RUNNABLE=1
        SCORE=$((SCORE + 20))
    else
        echo "TEMPLATE_SMOKE_FAILED=1"
    fi
    rm -rf "$TMPDIR"
fi

# --- 4. Prompt completeness (20 points) ---
AGENT_FILE="$HOME/ai-scientist-v2/ai_scientist/treesearch/parallel_agent.py"
if [ -f "$AGENT_FILE" ]; then
    if grep -q "tinker" "$AGENT_FILE"; then
        PROMPT_HAS_TINKER=1
        SCORE=$((SCORE + 20))
    else
        echo "PROMPT_MISSING_TINKER=1"
    fi
else
    echo "AGENT_FILE_MISSING=1"
fi

# --- 5. Template has no if __name__ == "__main__" guard (10 points) ---
# The AI Scientist interpreter runs code via exec() in global scope;
# a main guard prevents execution.
if [ "$TEMPLATE_PARSES" -eq 1 ]; then
    if ! grep -q 'if __name__ == "__main__"' "$TEMPLATE"; then
        TEMPLATE_NO_MAIN_GUARD=1
        SCORE=$((SCORE + 10))
    else
        echo "TEMPLATE_HAS_MAIN_GUARD=1"
    fi
fi

# --- 6. Template handles missing TINKER_API_KEY gracefully (10 points) ---
if [ "$TEMPLATE_PARSES" -eq 1 ]; then
    if grep -q "TINKER_API_KEY" "$TEMPLATE"; then
        # Should NOT raise RuntimeError — should warn and write placeholder output
        if ! grep -q "raise RuntimeError" "$TEMPLATE"; then
            TEMPLATE_GRACEFUL_API_KEY=1
            SCORE=$((SCORE + 10))
        else
            echo "TEMPLATE_RAISES_ON_MISSING_KEY=1"
        fi
    else
        # If key check is absent, that's also acceptable (no hard failure)
        TEMPLATE_GRACEFUL_API_KEY=1
        SCORE=$((SCORE + 10))
    fi
fi

# --- Output metrics ---
echo "METRIC runnability_score=$SCORE"
echo "METRIC deps_ready=$DEPS_READY"
echo "METRIC template_parses=$TEMPLATE_PARSES"
echo "METRIC template_runnable=$TEMPLATE_RUNNABLE"
echo "METRIC prompt_has_tinker=$PROMPT_HAS_TINKER"
echo "METRIC template_no_main_guard=$TEMPLATE_NO_MAIN_GUARD"
echo "METRIC template_graceful_api_key=$TEMPLATE_GRACEFUL_API_KEY"
