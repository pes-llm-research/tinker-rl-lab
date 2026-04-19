#!/bin/bash
set -euo pipefail
# Apply tinker-rl-lab integration patches to ~/ai-scientist-v2

SRC="/Users/arvind/paper/tinker-rl-lab/ai-scientist-v2-integration"
DST="$HOME/ai-scientist-v2"

# Copy modified parallel_agent.py to add tinker to agent environment prompt
if [ -f "$SRC/ai_scientist/treesearch/parallel_agent.py" ]; then
    cp "$SRC/ai_scientist/treesearch/parallel_agent.py" \
       "$DST/ai_scientist/treesearch/parallel_agent.py"
    echo "PATCHED parallel_agent.py"
fi

# Copy modified experiment templates
cp "$SRC/ai_scientist/ideas/tinker_grpo_rl.py" "$DST/ai_scientist/ideas/tinker_grpo_rl.py"
echo "PATCHED tinker_grpo_rl.py"

# Copy local TRL template (no Tinker API needed)
cp "$SRC/ai_scientist/ideas/trl_local_grpo.py" "$DST/ai_scientist/ideas/trl_local_grpo.py"
cp "$SRC/ai_scientist/ideas/trl_local_grpo.json" "$DST/ai_scientist/ideas/trl_local_grpo.json"
echo "PATCHED trl_local_grpo.py + trl_local_grpo.json"

# Copy tool-use reward design idea
cp "$SRC/ai_scientist/ideas/tool_use_reward_design.json" "$DST/ai_scientist/ideas/tool_use_reward_design.json"
echo "PATCHED tool_use_reward_design.json"

# Copy modified BFTS config
cp "$SRC/bfts_config.yaml" "$DST/bfts_config.yaml"
echo "PATCHED bfts_config.yaml"

echo "Patches applied."
